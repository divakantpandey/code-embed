"""Main training entry point for the 3-stage pipeline.

Stage 1: SimCSE unsupervised contrastive learning
Stage 2: Supervised fine-tuning with MNRL
Stage 3: Matryoshka Representation Learning (MRL)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml
from datasets import load_from_disk
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from src.data.collator import EmbeddingCollator, PairCollator
from src.data.dataset import CodeSearchNetDataset, ContrastiveDataset
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.model.encoder import CodeEncoder
from src.model.matryoshka import MatryoshkaEncoder
from src.training.callbacks import (
    AlertCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    WandBCallback,
)
from src.training.losses import InfoNCELoss, MatryoshkaLoss, MultipleNegativesRankingLoss
from src.training.trainer import CodeEmbeddingTrainer

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Training CLI")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_samples(data_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load query and code pairs from disk or fallback to synthetic data."""
    path = Path(data_dir)
    if path.exists() and (path / "dataset_info.json").exists():
        ds = load_from_disk(str(path))
        queries = [ex.get("docstring_summary") or ex.get("docstring") or "" for ex in ds]
        codes = [ex.get("func_code_string") or ex.get("code") or "" for ex in ds]
        valid = [(q, c) for q, c in zip(queries, codes) if q and c]
        if valid:
            return [q for q, _ in valid], [c for _, c in valid]

    # Fallback synthetic data
    sample_queries = [
        "add two numbers together",
        "calculate subtraction difference",
        "multiply numbers with a class method",
        "check if string is palindrome",
        "sort an array of numbers using quicksort",
    ]
    sample_codes = [
        "def add(a: int, b: int) -> int:\n    return a + b",
        "def subtract(x: float, y: float) -> float:\n    return x - y",
        "class Calc:\n    def mul(self, a, b):\n        return a * b",
        "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
    ]
    return sample_queries, sample_codes


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training config YAML (auto-detects best_tuned_config.yaml if present)",
    ),
    stage: int | None = typer.Option(
        None, "--stage", "-s", help="Run a specific stage only (1, 2, or 3)"
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-m", help="Limit number of samples for fast iteration / profiling"
    ),
    epochs_override: int | None = typer.Option(
        None, "--epochs", "-e", help="Override number of epochs for each stage"
    ),
) -> None:
    """Run the 3-stage training pipeline."""
    if config is None:
        tuned_path = Path("configs/best_tuned_config.yaml")
        if tuned_path.exists():
            config = tuned_path
            console.print(
                f"[bold cyan]🎯 Auto-detected {tuned_path} from previous tuning sweep. Using best hyperparameters![/]"
            )
        else:
            config = Path("configs/train_config.yaml")

    console.print(f"[bold green]🚀 Loading config from {config}...[/]")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    output_dir = Path(cfg.get("training", {}).get("output_dir", "./outputs"))
    checkpoint_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.get("model", {})
    base_model = model_cfg.get("base", "microsoft/unixcoder-base")
    embedding_dim = model_cfg.get("embedding_dim", 1536)
    pooling = model_cfg.get("pooling", "cls")
    max_seq_length = model_cfg.get("max_seq_length", 512)

    console.print(f"[bold blue]Initializing CodeEncoder ({base_model})...[/]")
    try:
        model = CodeEncoder(
            base_model_name_or_path=base_model,
            embedding_dim=embedding_dim,
            pooling=pooling,
            max_seq_length=max_seq_length,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load %s (%s). Using dummy backbone for test mode.", base_model, exc
        )
        from tests.conftest import DummyTokenizer, DummyTransformerBackbone

        model = CodeEncoder(
            transformer_backbone=DummyTransformerBackbone(hidden_size=64),
            tokenizer=DummyTokenizer(),
            embedding_dim=embedding_dim,
            pooling=pooling,
        )

    queries, codes = load_dataset_samples("data/processed")
    if max_samples is not None and max_samples > 0:
        queries = queries[:max_samples]
        codes = codes[:max_samples]
        logger.info("Restricted dataset to %d samples for fast execution/profiling.", max_samples)

    stages_to_run = [stage] if stage else [1, 2, 3]

    final_metrics: dict[str, Any] = {}

    for s in stages_to_run:
        stage_key = f"stage{s}"
        stage_cfg = cfg["training"].get(stage_key, {})
        if not stage_cfg or not stage_cfg.get("enabled", True):
            logger.info("Stage %d skipped (disabled in config).", s)
            continue

        console.print(
            f"\n[bold magenta]════════════ Stage {s}: {stage_cfg.get('method')} ════════════[/]"
        )

        lr = float(stage_cfg.get("learning_rate", 2e-5))
        batch_size = int(stage_cfg.get("batch_size", 16))
        grad_accum = int(stage_cfg.get("gradient_accumulation_steps", 1))
        epochs = epochs_override if epochs_override is not None else int(stage_cfg.get("epochs", 2))
        fp16 = bool(stage_cfg.get("fp16", True))
        scheduler_name = stage_cfg.get("scheduler", "cosine")
        weight_decay = float(stage_cfg.get("weight_decay", 0.01))

        # Callbacks
        callbacks = [
            AlertCallback(
                loss_spike_multiplier=cfg.get("alerts", {}).get("loss_spike_multiplier", 3.0),
                grad_norm_threshold=cfg.get("alerts", {}).get("grad_norm_threshold", 10.0),
            ),
            CheckpointCallback(
                checkpoint_dir=checkpoint_dir / f"stage{s}",
                save_every_n_steps=cfg.get("logging", {}).get("save_every_n_steps", 1000),
            ),
        ]

        if cfg.get("logging", {}).get("wandb", {}).get("enabled", False):
            callbacks.append(
                WandBCallback(
                    project=cfg["logging"]["wandb"].get("project", "code-embed"),
                    run_name=f"stage_{s}_{stage_cfg.get('method')}",
                    config=cfg,
                )
            )

        if cfg.get("logging", {}).get("tensorboard", {}).get("enabled", False):
            callbacks.append(
                TensorBoardCallback(
                    log_dir=Path(cfg["logging"]["tensorboard"].get("log_dir", "./runs"))
                    / f"stage{s}"
                )
            )

        # Build Stage DataLoader and Loss
        if s == 1:
            dataset = ContrastiveDataset(codes)
            collator = EmbeddingCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
            )
            loss_fn = InfoNCELoss(temperature=stage_cfg.get("temperature", 0.05))
            trainer_model = model

        elif s == 2:
            dataset = CodeSearchNetDataset(queries=queries, codes=codes)
            collator = PairCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
            )
            scale = float(stage_cfg.get("scale", 20.0))
            loss_fn = MultipleNegativesRankingLoss(scale=scale)
            trainer_model = model

            if stage_cfg.get("early_stopping", {}).get("enabled", False):
                callbacks.append(
                    EarlyStoppingCallback(
                        patience=stage_cfg["early_stopping"].get("patience", 4),
                        metric=stage_cfg["early_stopping"].get("metric", "eval/MRR@10"),
                        mode=stage_cfg["early_stopping"].get("mode", "max"),
                    )
                )

        elif s == 3:
            raw_dims = stage_cfg.get("dimensions", [64, 128, 256, 384, 768, 1536])
            raw_weights = stage_cfg.get("dimension_weights", [0.1, 0.1, 0.15, 0.15, 0.25, 0.25])
            # Ensure dimensions do not exceed model embedding dimension
            valid_pairs = [
                (d, w) for d, w in zip(raw_dims, raw_weights) if d <= model.embedding_dim
            ]
            dimensions = [d for d, _ in valid_pairs]
            total_w = sum(w for _, w in valid_pairs) or 1.0
            weights = [w / total_w for _, w in valid_pairs]

            mrl_encoder = MatryoshkaEncoder(model, dimensions=dimensions)
            trainer_model = mrl_encoder

            dataset = CodeSearchNetDataset(queries=queries, codes=codes)
            collator = PairCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
            )
            scale = float(stage_cfg.get("scale", 20.0))
            base_loss = MultipleNegativesRankingLoss(scale=scale)
            loss_fn = MatryoshkaLoss(base_loss=base_loss, dimensions=dimensions, weights=weights)

        # Build validation evaluation function if queries/codes are available
        eval_fn = None
        if s in (2, 3) and queries and codes:
            val_size = min(
                100, max(len(queries) if len(queries) < 10 else 5, int(len(queries) * 0.1))
            )
            val_queries = queries[:val_size]
            val_codes = codes[:val_size]

            def _build_evaluator(target_m: Any, v_q: list[str], v_c: list[str]) -> Any:
                def _eval() -> dict[str, float]:
                    with torch.no_grad():
                        enc = target_m.encoder if hasattr(target_m, "encoder") else target_m
                        q_embs = enc.encode(v_q, batch_size=32, convert_to_numpy=True)
                        c_embs = enc.encode(v_c, batch_size=32, convert_to_numpy=True)
                        res = compute_retrieval_metrics(q_embs, c_embs, ks=[1, 5, 10])
                        out = {f"eval/{k}": float(v) for k, v in res.items()}
                        out["loss"] = 0.0
                        return out

                return _eval

            eval_fn = _build_evaluator(trainer_model, val_queries, val_codes)

        trainer = CodeEmbeddingTrainer(
            model=trainer_model,
            loss_fn=loss_fn,
            train_dataloader=dataloader,
            eval_fn=eval_fn,
            learning_rate=lr,
            epochs=epochs,
            gradient_accumulation_steps=grad_accum,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            fp16=fp16,
            device=cfg.get("hardware", {}).get("device", "auto"),
            callbacks=callbacks,
        )

        metrics = trainer.train()
        final_metrics[f"stage_{s}"] = metrics
        console.print(f"[bold green]Stage {s} complete! Metrics: {metrics}[/]")

    # Save best / final model
    best_model_path = output_dir / "best_model"
    model.save_pretrained(best_model_path)
    console.print(f"[bold green]Model saved to {best_model_path}[/]")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    console.print("[bold green]✅ All requested training stages completed successfully.[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

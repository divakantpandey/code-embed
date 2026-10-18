"""Automated hyperparameter tuning script using Optuna.

Systematically explores hyperparameters (learning rates, loss scale/temperature,
weight decay, batch sizes, gradient accumulation, warmup ratios, and pooling)
to maximize code retrieval accuracy (MRR@10) with trial pruning and automatic
best configuration generation.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch.utils.data import DataLoader

from src.data.collator import PairCollator
from src.data.dataset import CodeSearchNetDataset
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.model.encoder import CodeEncoder
from src.model.matryoshka import MatryoshkaEncoder
from src.training.losses import InfoNCELoss, MatryoshkaLoss, MultipleNegativesRankingLoss
from src.training.trainer import CodeEmbeddingTrainer

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Hyperparameter Tuning CLI")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across packages."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_samples(data_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load query and code pairs from disk or fallback to synthetic data."""
    path = Path(data_dir)
    if path.exists() and (path / "dataset_info.json").exists():
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        queries = [ex.get("docstring_summary") or ex.get("docstring") or "" for ex in ds]
        codes = [ex.get("func_code_string") or ex.get("code") or "" for ex in ds]
        valid = [(q, c) for q, c in zip(queries, codes) if q and c]
        if valid:
            return [q for q, _ in valid], [c for _, c in valid]

    # Synthetic fallback data
    sample_queries = [
        "add two numbers together",
        "calculate subtraction difference",
        "multiply numbers with a class method",
        "check if string is palindrome",
        "sort an array of numbers using quicksort",
        "compute factorial recursively",
        "find maximum element in list",
        "reverse a singly linked list",
    ]
    sample_codes = [
        "def add(a: int, b: int) -> int:\n    return a + b",
        "def subtract(x: float, y: float) -> float:\n    return x - y",
        "class Calc:\n    def mul(self, a, b):\n        return a * b",
        "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
        "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "def find_max(nums: list[int]) -> int:\n    return max(nums) if nums else 0",
        "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev",
    ]
    return sample_queries, sample_codes


def evaluate_retrieval(
    model: CodeEncoder | MatryoshkaEncoder,
    queries: list[str],
    codes: list[str],
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate model on queries and codes using MRR@10 and Recall@1."""
    with torch.no_grad():
        enc = model.encoder if hasattr(model, "encoder") else model
        q_embs = enc.encode(queries, batch_size=batch_size, convert_to_numpy=True)
        c_embs = enc.encode(codes, batch_size=batch_size, convert_to_numpy=True)
        return compute_retrieval_metrics(q_embs, c_embs, ks=[1, 5, 10])


def create_encoder(
    base_model: str,
    embedding_dim: int,
    pooling: str,
    max_seq_length: int,
) -> CodeEncoder:
    """Instantiate CodeEncoder with fallback to dummy backbone in test environments."""
    try:
        return CodeEncoder(
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

        return CodeEncoder(
            transformer_backbone=DummyTransformerBackbone(hidden_size=64),
            tokenizer=DummyTokenizer(),
            embedding_dim=embedding_dim,
            pooling=pooling,
        )


def run_trial(
    trial: optuna.Trial,
    base_cfg: dict[str, Any],
    queries: list[str],
    codes: list[str],
    stage_to_tune: int,
    epochs_per_trial: int,
    device: str,
    max_eval_samples: int,
) -> float:
    """Execute a single hyperparameter optimization trial."""
    model_cfg = base_cfg.get("model", {})
    base_model = model_cfg.get("base", "microsoft/unixcoder-base")
    embedding_dim = model_cfg.get("embedding_dim", 1536)
    max_seq_length = model_cfg.get("max_seq_length", 512)

    # Suggest hyperparameters
    pooling = trial.suggest_categorical("pooling", ["cls", "mean"])

    if stage_to_tune == 1:
        lr = trial.suggest_float("stage1_lr", 1.0e-5, 8.0e-5, log=True)
        weight_decay = trial.suggest_float("stage1_weight_decay", 1.0e-4, 1.0e-1, log=True)
        batch_size = trial.suggest_categorical("stage1_batch_size", [8, 16, 32])
        grad_accum = trial.suggest_categorical("stage1_grad_accum", [2, 4, 8])
        temperature = trial.suggest_float("stage1_temperature", 0.02, 0.10, step=0.01)
        loss_fn: torch.nn.Module = InfoNCELoss(temperature=temperature)
    elif stage_to_tune == 3:
        lr = trial.suggest_float("stage3_lr", 5.0e-6, 3.0e-5, log=True)
        weight_decay = trial.suggest_float("stage3_weight_decay", 1.0e-4, 1.0e-1, log=True)
        batch_size = trial.suggest_categorical("stage3_batch_size", [8, 16, 32])
        grad_accum = trial.suggest_categorical("stage3_grad_accum", [2, 4, 8])
        scale = trial.suggest_float("stage3_scale", 15.0, 30.0, step=2.5)
        base_loss = MultipleNegativesRankingLoss(scale=scale)
        dimensions = (
            base_cfg.get("training", {})
            .get("stage3", {})
            .get("dimensions", [64, 128, 256, 384, 768, 1536, 3072])
        )
        weights = (
            base_cfg.get("training", {})
            .get("stage3", {})
            .get("dimension_weights", [0.1, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2])
        )
        loss_fn = MatryoshkaLoss(base_loss=base_loss, dimensions=dimensions, weights=weights)
    else:  # Stage 2 (MNRL, primary retrieval fine-tuning)
        lr = trial.suggest_float("stage2_lr", 5.0e-6, 5.0e-5, log=True)
        scale = trial.suggest_float("stage2_scale", 15.0, 30.0, step=2.5)
        weight_decay = trial.suggest_float("stage2_weight_decay", 1.0e-4, 1.0e-1, log=True)
        batch_size = trial.suggest_categorical("stage2_batch_size", [8, 16, 32])
        grad_accum = trial.suggest_categorical("stage2_grad_accum", [2, 4, 8])
        loss_fn = MultipleNegativesRankingLoss(scale=scale)

    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.20, step=0.05)
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["cosine", "cosine_with_restarts", "linear"]
    )

    # Initialize model
    model = create_encoder(
        base_model=base_model,
        embedding_dim=embedding_dim,
        pooling=pooling,
        max_seq_length=max_seq_length,
    )

    # Split dataset into train and validation
    val_size = min(max_eval_samples, max(2, int(len(queries) * 0.2)))
    val_queries = queries[:val_size]
    val_codes = codes[:val_size]
    train_queries = queries[val_size:] if len(queries) > val_size else queries
    train_codes = codes[val_size:] if len(codes) > val_size else codes

    dataset = CodeSearchNetDataset(queries=train_queries, codes=train_codes)
    collator = PairCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collator,
    )

    trainer_model: CodeEncoder | MatryoshkaEncoder = model
    if stage_to_tune == 3:
        dimensions = (
            base_cfg.get("training", {})
            .get("stage3", {})
            .get("dimensions", [64, 128, 256, 384, 768, 1536, 3072])
        )
        trainer_model = MatryoshkaEncoder(model, dimensions=dimensions)

    best_mrr = 0.0

    for epoch in range(epochs_per_trial):
        trainer = CodeEmbeddingTrainer(
            model=trainer_model,
            loss_fn=loss_fn,
            train_dataloader=dataloader,
            learning_rate=lr,
            epochs=1,
            gradient_accumulation_steps=grad_accum,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            scheduler_name=scheduler_name,
            fp16=device != "cpu",
            device=device,
        )
        trainer.train()

        # Evaluate validation MRR@10
        eval_metrics = evaluate_retrieval(trainer_model, val_queries, val_codes)
        val_mrr = eval_metrics.get("MRR@10", 0.0)
        best_mrr = max(best_mrr, val_mrr)

        # Report to Optuna and check for pruning
        trial.report(val_mrr, step=epoch)
        if trial.should_prune():
            logger.info(
                "Trial %d pruned at epoch %d with MRR@10=%.4f", trial.number, epoch, val_mrr
            )
            raise optuna.TrialPruned()

    return best_mrr


@app.command()
def tune(
    config: Path = typer.Option(
        "configs/train_config.yaml", "--config", "-c", help="Path to base train configuration"
    ),
    n_trials: int = typer.Option(
        15, "--n-trials", "-n", help="Total number of hyperparameter trials to run"
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", "-t", help="Timeout in seconds for hyperparameter search"
    ),
    stage: int = typer.Option(
        2, "--stage", "-s", help="Training stage to focus tuning on (1, 2, or 3)"
    ),
    epochs_per_trial: int = typer.Option(
        3, "--epochs-per-trial", "-e", help="Number of training epochs per trial"
    ),
    val_samples: int = typer.Option(
        100, "--val-samples", help="Number of held-out validation samples for fast evaluation"
    ),
    output_config: Path = typer.Option(
        "configs/best_tuned_config.yaml",
        "--output-config",
        "-o",
        help="Path to write the best-tuned configuration YAML",
    ),
    results_json: Path = typer.Option(
        "outputs/tuning/tuning_results.json",
        "--results-json",
        "-r",
        help="Path to save trial results summary JSON",
    ),
    study_name: str = typer.Option(
        "code-embed-tune", "--study-name", help="Name of Optuna study"
    ),
    storage: str | None = typer.Option(
        "sqlite:///outputs/tuning/optuna_study.db",
        "--storage",
        help="Database URI for persisting trials across runs (pass '' for in-memory)",
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Compute device ('auto', 'cpu', 'mps', 'cuda')"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for search reproducibility"),
) -> None:
    """Automate hyperparameter optimization using Optuna to find the best configuration."""
    console.print(f"[bold green]🔎 Initializing hyperparameter search for Stage {stage}...[/]")
    set_seed(seed)

    with open(config) as f:
        base_cfg = yaml.safe_load(f)

    # Resolve target device
    target_device = device
    if target_device == "auto":
        target_device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

    queries, codes = load_dataset_samples("data/processed")
    console.print(f"Loaded {len(queries)} samples for training and validation.")

    # Create or resume persistent Optuna study
    storage_url = storage if storage and storage.strip() else None
    if storage_url and storage_url.startswith("sqlite:///"):
        db_path = Path(storage_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    console.print(
        f"[bold blue]Active Optuna study '{study_name}' ({len(study.trials)} previous trials found). Running {n_trials} new trials...[/]"
    )

    def _objective(trial: optuna.Trial) -> float:
        return run_trial(
            trial=trial,
            base_cfg=base_cfg,
            queries=queries,
            codes=codes,
            stage_to_tune=stage,
            epochs_per_trial=epochs_per_trial,
            device=target_device,
            max_eval_samples=val_samples,
        )

    study.optimize(_objective, n_trials=n_trials, timeout=timeout)

    # Results summary
    console.print("\n[bold green]════════════ Tuning Results Summary ════════════[/]")
    console.print(f"[bold cyan]Best Trial: #{study.best_trial.number}[/]")
    console.print(f"[bold cyan]Best Validation MRR@10: {study.best_value:.4f}[/]")

    # Build results table
    table = Table(title="Top Tuning Trials")
    table.add_column("Trial #", justify="center", style="cyan")
    table.add_column("MRR@10", justify="right", style="green")
    table.add_column("State", justify="center")
    table.add_column("Key Parameters", justify="left")

    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -1.0,
        reverse=True,
    )

    for t in sorted_trials[:10]:
        val_str = f"{t.value:.4f}" if t.value is not None else "N/A"
        state_str = (
            "[green]COMPLETE[/]"
            if t.state == optuna.trial.TrialState.COMPLETE
            else f"[yellow]{t.state.name}[/]"
        )
        params_summary = ", ".join(
            f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(t.params.items())[:4]
        )
        table.add_row(str(t.number), val_str, state_str, params_summary)

    console.print(table)

    # Generate best config
    best_cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
    best_params = study.best_params

    # Update model settings
    if "pooling" in best_params:
        best_cfg.setdefault("model", {})["pooling"] = best_params["pooling"]

    # Update stage settings
    stage_key = f"stage{stage}"
    stage_section = best_cfg.setdefault("training", {}).setdefault(stage_key, {})

    for param_name, param_val in best_params.items():
        if param_name.startswith(f"stage{stage}_"):
            clean_name = param_name.removeprefix(f"stage{stage}_")
            if clean_name == "grad_accum":
                stage_section["gradient_accumulation_steps"] = int(param_val)
            elif clean_name == "lr":
                stage_section["learning_rate"] = float(param_val)
            else:
                stage_section[clean_name] = param_val
        elif param_name == "warmup_ratio":
            stage_section["warmup_ratio"] = float(param_val)
        elif param_name == "scheduler":
            stage_section["scheduler"] = str(param_val)

    # Save best configuration YAML
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    console.print(f"[bold green]💾 Saved best tuned configuration to {output_config}[/]")

    # Save results JSON
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "study_name": study_name,
        "best_trial_number": study.best_trial.number,
        "best_mrr10": study.best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    console.print(f"[bold green]📊 Saved full trial results to {results_json}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

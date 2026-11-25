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
from datasets import Dataset, concatenate_datasets, load_from_disk
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from src.data.collator import EmbeddingCollator, PairCollator
from src.data.dataset import CodeSearchNetDataset, ContrastiveDataset, TripletDataset
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
from src.training.checks import PreflightChecker
from src.training.losses import InfoNCELoss, MatryoshkaLoss, MultipleNegativesRankingLoss
from src.training.trainer import CodeEmbeddingTrainer

load_dotenv()

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="CodeEmbed Training CLI")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_multilingual_dataset_samples(
    data_dir: str | Path,
    target_languages: list[str] | None = None,
    full_scale: bool = False,
    include_cosqa: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Load query, code, and language triples from disk or fallback to synthetic data."""
    path = Path(data_dir)
    all_queries: list[str] = []
    all_codes: list[str] = []
    all_langs: list[str] = []

    # Priority 1: Check if full preprocessed directory exists (e.g. data/processed/full)
    full_subdir = path / "full" if path.name != "full" else path
    if (full_subdir / "dataset_info.json").exists() or (full_subdir / "state.json").exists():
        try:
            full_ds = load_from_disk(str(full_subdir))
            doc_col = (
                "func_documentation_string"
                if "func_documentation_string" in full_ds.column_names
                else "docstring_summary"
            )
            code_col = "func_code_string" if "func_code_string" in full_ds.column_names else "code"
            lang_col = "language" if "language" in full_ds.column_names else None

            if target_languages:
                if lang_col:
                    filtered_ds = full_ds.filter(
                        lambda ex: (
                            str(ex.get(lang_col, "python")).lower().strip() in target_languages
                        )
                    )
                else:
                    filtered_ds = full_ds if "python" in target_languages else None
            else:
                filtered_ds = full_ds

            if filtered_ds is not None and len(filtered_ds) > 0:
                all_queries = list(filtered_ds[doc_col])
                all_codes = list(filtered_ds[code_col])
                all_langs = (
                    list(filtered_ds[lang_col]) if lang_col else ["python"] * len(all_queries)
                )
                logger.info(
                    "Loaded %d samples directly from full dataset at %s",
                    len(all_queries),
                    full_subdir,
                )
                return all_queries, all_codes, all_langs
        except Exception as exc:
            logger.debug("Could not load from full dataset dir %s: %s", full_subdir, exc)

    # Priority 2: Check for multi-shard Arrow files (e.g. data-0000X-of-00004.arrow)
    shard_files = sorted(path.glob("data-*-of-*.arrow"))
    if not all_queries and len(shard_files) > 1:
        try:
            shards = [Dataset.from_file(str(sf)) for sf in shard_files]
            combined_arrow = concatenate_datasets(shards)
            doc_col = (
                "func_documentation_string"
                if "func_documentation_string" in combined_arrow.column_names
                else "docstring_summary"
            )
            code_col = (
                "func_code_string" if "func_code_string" in combined_arrow.column_names else "code"
            )
            lang_col = "language" if "language" in combined_arrow.column_names else None

            if target_languages and lang_col:
                combined_arrow = combined_arrow.filter(
                    lambda ex: str(ex.get(lang_col, "python")).lower().strip() in target_languages
                )

            all_queries = list(combined_arrow[doc_col])
            all_codes = list(combined_arrow[code_col])
            all_langs = (
                list(combined_arrow[lang_col]) if lang_col else ["python"] * len(all_queries)
            )
            logger.info(
                "Loaded %d samples from %d Arrow shards in %s",
                len(all_queries),
                len(shard_files),
                path,
            )
            return all_queries, all_codes, all_langs
        except Exception as exc:
            logger.debug("Could not load from arrow shards: %s", exc)

    # Priority 3: Check if language-specific subdirectories exist
    if target_languages:
        for lang in target_languages:
            sub = path / lang
            if sub.exists() and (sub / "dataset_info.json").exists():
                try:
                    ds = load_from_disk(str(sub))
                    doc_col = (
                        "func_documentation_string"
                        if "func_documentation_string" in ds.column_names
                        else "docstring_summary"
                    )
                    code_col = (
                        "func_code_string" if "func_code_string" in ds.column_names else "code"
                    )
                    all_queries.extend(ds[doc_col])
                    all_codes.extend(ds[code_col])
                    all_langs.extend([lang] * len(ds))
                except Exception:
                    pass

    # Priority 4: Check main directory
    if not all_queries and path.exists() and (path / "dataset_info.json").exists():
        try:
            ds = load_from_disk(str(path))
            for ex in ds:
                q = (
                    ex.get("docstring_summary")
                    or ex.get("func_documentation_string")
                    or ex.get("docstring")
                    or ""
                ).strip()
                c = (ex.get("func_code_string") or ex.get("code") or "").strip()
                lang_tag = str(ex.get("language") or "python").lower().strip()
                if (target_languages is None or lang_tag in target_languages) and q and c:
                    all_queries.append(q)
                    all_codes.append(c)
                    all_langs.append(lang_tag)
        except Exception:
            pass

    # Priority 5: If full_scale requested and sample count is small, load directly from data/raw/train
    raw_path = Path("data/raw")
    if (full_scale or len(all_queries) < 100) and (raw_path / "train").exists():
        try:
            raw_ds = load_from_disk(str(raw_path))
            train_part = raw_ds.get("train", raw_ds)
            doc_col = (
                "func_documentation_string"
                if "func_documentation_string" in train_part.column_names
                else "docstring"
            )
            code_col = (
                "func_code_string" if "func_code_string" in train_part.column_names else "code"
            )
            all_queries = list(train_part[doc_col])
            all_codes = list(train_part[code_col])
            all_langs = (
                list(train_part["language"])
                if "language" in train_part.column_names
                else ["python"] * len(all_queries)
            )
            logger.info(
                "Loaded %d samples directly from raw train split (%s)", len(all_queries), raw_path
            )
            return all_queries, all_codes, all_langs
        except Exception as exc:
            logger.debug("Could not fallback to data/raw: %s", exc)

    # Optional CoSQA integration for boosted retrieval accuracy
    if include_cosqa and (target_languages is None or "python" in target_languages):
        try:
            from datasets import load_dataset as hf_load_dataset

            cosqa_ds = hf_load_dataset("gonglinyuan/CoSQA", split="train")
            pos_cosqa = cosqa_ds.filter(lambda x: x.get("label", 1) == 1)
            cosqa_q = list(pos_cosqa["query"])
            cosqa_c = list(pos_cosqa["code"])
            all_queries.extend(cosqa_q)
            all_codes.extend(cosqa_c)
            all_langs.extend(["python"] * len(cosqa_q))
            logger.info("Added %d CoSQA natural language query pairs", len(cosqa_q))
        except Exception as cosqa_err:
            logger.debug("CoSQA dataset not available online: %s", cosqa_err)

    if all_queries and all_codes:
        return all_queries, all_codes, all_langs

    # Fallback synthetic data across languages
    from scripts.download_data import SYNTHETIC_DATA

    langs_to_use = target_languages or ["python", "go", "java", "ruby", "php", "javascript"]
    fb_q, fb_c, fb_l = [], [], []
    for lang_tag in langs_to_use:
        s = SYNTHETIC_DATA.get(lang_tag, SYNTHETIC_DATA["python"])
        for q, c in zip(s["docstring_summary"], s["func_code_string"]):
            fb_q.append(q.strip())
            fb_c.append(c.strip())
            fb_l.append(lang_tag)
    return fb_q, fb_c, fb_l


def load_dataset_samples(data_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load query and code pairs from disk or fallback to synthetic data."""
    q, c, _ = load_multilingual_dataset_samples(data_dir)
    return q, c


@app.command()
def train(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training config YAML (auto-detects full_scale_config.yaml or best_tuned_config.yaml)",
    ),
    stage: int | None = typer.Option(
        None, "--stage", "-s", help="Run a specific stage only (1, 2, or 3)"
    ),
    languages: str | None = typer.Option(
        None,
        "--languages",
        "-l",
        help="Comma-separated languages to train on (e.g. 'go,java,ruby,php,javascript,python')",
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-m", help="Limit number of samples for fast iteration / profiling"
    ),
    epochs_override: int | None = typer.Option(
        None, "--epochs", "-e", help="Override number of epochs for each stage"
    ),
    full_scale: bool = typer.Option(
        False,
        "--full-scale",
        "-F",
        help="Run full-scale training on complete dataset with GPU acceleration and optimal hyperparameters",
    ),
    resume_from_checkpoint: Path | None = typer.Option(
        None,
        "--resume-from-checkpoint",
        "-r",
        help="Resume training from an existing checkpoint directory",
    ),
    push_to_hub: bool | None = typer.Option(
        None,
        "--push-to-hub/--no-push-to-hub",
        help="Push checkpoints and final model to Hugging Face Hub",
    ),
    hub_repo_id: str | None = typer.Option(
        None,
        "--hub-repo-id",
        help="Hugging Face repository ID (e.g. 'divakantpandey/code-embed')",
    ),
    precision: str | None = typer.Option(
        None,
        "--precision",
        "-p",
        help="Mixed precision mode: 'bf16', 'fp16', or 'fp32' (defaults to config or auto-detect)",
    ),
    init_model: Path | None = typer.Option(
        None,
        "--init-model",
        help="Initialize model weights from an existing model directory (e.g. previous stage last_model)",
    ),
    preflight_only: bool = typer.Option(
        False,
        "--preflight-only",
        help="Run system and hardware pre-flight checks and exit without training",
    ),
) -> None:
    """Run the 3-stage training pipeline (with optional full-scale remote RunPod execution)."""
    if config is None:
        full_config_path = Path("configs/full_scale_config.yaml")
        tuned_path = Path("configs/best_tuned_config.yaml")
        if full_scale and full_config_path.exists():
            config = full_config_path
            console.print(
                f"[bold cyan]🎯 Full-scale mode detected: Loading RunPod configuration from {full_config_path}[/]"
            )
        elif tuned_path.exists():
            config = tuned_path
            console.print(
                f"[bold cyan]🎯 Auto-detected {tuned_path} from previous tuning sweep. Using best hyperparameters![/]"
            )
        else:
            config = Path("configs/train_config.yaml")

    console.print(f"[bold green]🚀 Loading config from {config}...[/]")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    # If this is a language preset config without a "training" section, overlay onto configs/train_config.yaml
    if "training" not in cfg:
        base_train_config = Path("configs/train_config.yaml")
        if base_train_config.exists():
            with open(base_train_config) as f:
                base_cfg = yaml.safe_load(f)
            base_cfg.update(cfg)
            cfg = base_cfg

    # Hugging Face Checkpointing & Hub configuration
    hf_ckpt_cfg = cfg.get("hf_checkpointing", {})
    effective_push_to_hub = (
        push_to_hub if push_to_hub is not None else bool(hf_ckpt_cfg.get("push_to_hub", False))
    )
    effective_hub_repo = (
        hub_repo_id or hf_ckpt_cfg.get("hub_repo_id") or "divakantpandey/code-embed-v2"
    )
    effective_hub_private = bool(hf_ckpt_cfg.get("hub_private", False))

    # Preflight system and health checks
    preflight_report = PreflightChecker.run_all(
        config=cfg,
        full_scale=full_scale,
        require_hf_auth=effective_push_to_hub,
    )
    preflight_report.print_summary()

    if preflight_only:
        console.print("[bold cyan]Exiting as --preflight-only was requested.[/]")
        return

    if full_scale and not preflight_report.passed:
        console.print(
            "[bold red]❌ Cannot proceed with full-scale training due to preflight check failures. "
            "Please address the errors above before launching full-scale runs.[/]"
        )
        raise SystemExit(1)

    # Determine languages
    if languages:
        target_langs = [item.strip().lower() for item in languages.split(",") if item.strip()]
    elif "languages" in cfg:
        target_langs = [item.strip().lower() for item in cfg["languages"]]
    elif "language" in cfg and cfg["language"] != "multilingual":
        target_langs = [cfg["language"].strip().lower()]
    else:
        target_langs = None

    if target_langs:
        console.print(f"[bold cyan]Target training languages: {', '.join(target_langs)}[/]")

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

    init_model_path: Path | None = None
    if init_model is not None and Path(init_model).exists():
        init_model_path = Path(init_model)
    else:
        # Check if earlier stage has already been completed
        first_stage = stage if stage else (2 if not cfg.get("training", {}).get("stage1", {}).get("enabled", True) else 1)
        if first_stage > 1:
            for prev_s in range(first_stage - 1, 0, -1):
                prev_dir = checkpoint_dir / f"stage{prev_s}"
                for candidate in ["last_model", "best_model"]:
                    if (prev_dir / candidate).exists():
                        init_model_path = prev_dir / candidate
                        break
                if init_model_path:
                    break

    if init_model_path:
        console.print(
            f"[bold green]✨ Initializing model weights from previous stage: {init_model_path}[/]"
        )
        model = CodeEncoder.from_pretrained(str(init_model_path))
    else:
        console.print(f"[bold blue]Initializing CodeEncoder ({base_model}, pooling={pooling})...[/]")
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

    # Load dataset
    queries, codes, sample_langs = load_multilingual_dataset_samples(
        cfg.get("data", {}).get("data_dir", "data/processed"),
        target_languages=target_langs,
        full_scale=full_scale,
        include_cosqa=bool(
            full_scale
            or any(d.get("name") == "cosqa" for d in cfg.get("data", {}).get("datasets", []))
        ),
    )
    if max_samples is not None and max_samples > 0:
        queries = queries[:max_samples]
        codes = codes[:max_samples]
        sample_langs = sample_langs[:max_samples]
        logger.info("Restricted dataset to %d samples for fast execution/profiling.", max_samples)

    console.print(
        f"[bold green]Loaded dataset: {len(queries):,} total samples across languages.[/]"
    )

    # Train / validation split
    val_ratio = 0.05 if full_scale and len(queries) > 10000 else 0.1
    if queries and codes:
        val_size = (
            min(1000, max(2, int(len(queries) * val_ratio)))
            if len(queries) > 2
            else min(1, len(queries))
        )
        val_queries = queries[:val_size]
        val_codes = codes[:val_size]
        val_langs = sample_langs[:val_size]
        train_queries = queries[val_size:] if len(queries) > val_size else queries
        train_codes = codes[val_size:] if len(codes) > val_size else codes
        _train_langs = sample_langs[val_size:] if len(sample_langs) > val_size else sample_langs
    else:
        val_queries, val_codes, val_langs, train_queries, train_codes, _train_langs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

    # Hardware DataLoader options
    hw_cfg = cfg.get("hardware", {})
    num_workers = int(hw_cfg.get("num_workers", 0)) if torch.cuda.is_available() else 0
    pin_memory = bool(hw_cfg.get("pin_memory", True) and torch.cuda.is_available())
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(hw_cfg.get("prefetch_factor", 4))

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
        scheduler_name = stage_cfg.get("scheduler", "cosine")
        weight_decay = float(stage_cfg.get("weight_decay", 0.01))

        # Precision handling: CLI flag overrides config
        if precision == "bf16":
            bf16 = True
            fp16 = False
        elif precision == "fp16":
            bf16 = False
            fp16 = True
        elif precision == "fp32":
            bf16 = False
            fp16 = False
        else:
            hw_prec = str(hw_cfg.get("precision", "")).lower()
            bf16 = bool(stage_cfg.get("bf16", False) or hw_prec == "bf16")
            fp16 = bool(stage_cfg.get("fp16", True) and not bf16)

        # Callbacks
        stage_ckpt_dir = checkpoint_dir / f"stage{s}"
        callbacks = [
            AlertCallback(
                loss_spike_multiplier=cfg.get("alerts", {}).get("loss_spike_multiplier", 3.0),
                grad_norm_threshold=cfg.get("alerts", {}).get("grad_norm_threshold", 10.0),
                min_loss_threshold=cfg.get("alerts", {}).get("min_loss_threshold", 0.01),
            ),
            CheckpointCallback(
                checkpoint_dir=stage_ckpt_dir,
                save_every_n_steps=cfg.get("logging", {}).get("save_every_n_steps", 1000),
                push_to_hub=effective_push_to_hub,
                hub_repo_id=effective_hub_repo,
                hub_private=effective_hub_private,
            ),
        ]

        if cfg.get("logging", {}).get("wandb", {}).get("enabled", False):
            lang_label = cfg.get("language", "multilingual" if target_langs else "python")
            callbacks.append(
                WandBCallback(
                    project=cfg["logging"]["wandb"].get("project", "code-embed"),
                    run_name=f"v0.2.0_{lang_label}_stage_{s}_{stage_cfg.get('method')}",
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

        eval_dataloader: DataLoader[Any] | None = None
        eval_fn = None

        # Build Stage DataLoader and Loss
        if s == 1:
            dataset = ContrastiveDataset(train_codes)
            collator = EmbeddingCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                **loader_kwargs,
            )
            if val_codes:
                val_dataset = ContrastiveDataset(val_codes)
                eval_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    **loader_kwargs,
                )
            loss_fn = InfoNCELoss(temperature=stage_cfg.get("temperature", 0.05))
            trainer_model = model

        elif s == 2:
            # Stage 2: Supervised Fine-Tuning with MNRL and optional BM25 hard negatives
            hard_neg_cfg = cfg.get("data", {}).get("hard_negatives", {})
            use_hard_neg = (
                hard_neg_cfg.get("enabled", False) or hard_neg_cfg.get("bm25_top_k", 0) > 0
            )
            negatives = None

            if use_hard_neg and len(train_codes) > 10:
                console.print(
                    "[bold blue]⛏️ Mining BM25 hard negatives for Stage 2 contrastive fine-tuning...[/]"
                )
                try:
                    from src.data.hard_negatives import BM25NegativeMiner

                    miner = BM25NegativeMiner(
                        corpus=train_codes,
                        top_k=int(hard_neg_cfg.get("bm25_top_k", 50)),
                        min_rank=int(hard_neg_cfg.get("min_rank", 1)),
                    )
                    neg_limit = min(len(train_queries), 50000)
                    logger.info("Mining BM25 hard negatives for %d training queries...", neg_limit)
                    mined_negatives = miner.mine_negatives_batch(train_queries[:neg_limit])
                    if len(mined_negatives) < len(train_queries):
                        cycle_len = len(mined_negatives)
                        negatives = [
                            mined_negatives[i % cycle_len] for i in range(len(train_queries))
                        ]
                    else:
                        negatives = mined_negatives
                    console.print(
                        f"[bold green]✅ BM25 hard negatives mined ({len(negatives):,} samples)![/]"
                    )
                except Exception as neg_err:  # noqa: BLE001
                    logger.warning(
                        "Could not mine BM25 hard negatives (%s). Proceeding with in-batch negatives.",
                        neg_err,
                    )

            if negatives is not None:
                dataset = TripletDataset(
                    queries=train_queries, positives=train_codes, negatives=negatives
                )
            else:
                dataset = CodeSearchNetDataset(queries=train_queries, codes=train_codes)

            collator = PairCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                **loader_kwargs,
            )
            if val_queries and val_codes:
                val_dataset = CodeSearchNetDataset(queries=val_queries, codes=val_codes)
                eval_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    **loader_kwargs,
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

            dataset = CodeSearchNetDataset(queries=train_queries, codes=train_codes)
            collator = PairCollator(tokenizer=model.tokenizer, max_length=max_seq_length)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                **loader_kwargs,
            )
            if val_queries and val_codes:
                val_dataset = CodeSearchNetDataset(queries=val_queries, codes=val_codes)
                eval_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    **loader_kwargs,
                )
            scale = float(stage_cfg.get("scale", 20.0))
            base_loss = MultipleNegativesRankingLoss(scale=scale)
            loss_fn = MatryoshkaLoss(base_loss=base_loss, dimensions=dimensions, weights=weights)

        # Build validation evaluation function if queries/codes are available
        if s in (2, 3) and val_queries and val_codes:

            def _build_evaluator(
                target_m: Any, v_q: list[str], v_c: list[str], v_l: list[str]
            ) -> Any:
                def _eval() -> dict[str, float]:
                    with torch.no_grad():
                        enc = target_m.encoder if hasattr(target_m, "encoder") else target_m
                        q_embs = enc.encode(v_q, batch_size=32, convert_to_numpy=True)
                        c_embs = enc.encode(v_c, batch_size=32, convert_to_numpy=True)
                        res = compute_retrieval_metrics(q_embs, c_embs, ks=[1, 5, 10])
                        out_metrics = {f"eval/{k}": float(v) for k, v in res.items()}
                        # Compute per-language metrics if diverse
                        unique_langs = sorted(set(v_l))
                        if len(unique_langs) > 1:
                            for ul in unique_langs:
                                idxs = [i for i, val_l in enumerate(v_l) if val_l == ul]
                                if len(idxs) >= 2:
                                    sub_res = compute_retrieval_metrics(
                                        q_embs[idxs], c_embs[idxs], ks=[1, 5, 10]
                                    )
                                    for k, v in sub_res.items():
                                        out_metrics[f"eval/{ul}/{k}"] = float(v)
                        return out_metrics

                return _eval

            eval_fn = _build_evaluator(trainer_model, val_queries, val_codes, val_langs)

        trainer = CodeEmbeddingTrainer(
            model=trainer_model,
            loss_fn=loss_fn,
            train_dataloader=dataloader,
            eval_dataloader=eval_dataloader,
            eval_fn=eval_fn,
            learning_rate=lr,
            epochs=epochs,
            gradient_accumulation_steps=grad_accum,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            fp16=fp16,
            bf16=bf16,
            device=cfg.get("hardware", {}).get("device", "auto"),
            callbacks=callbacks,
        )

        # Resume from checkpoint if specified
        if resume_from_checkpoint is not None and Path(resume_from_checkpoint).exists():
            console.print(
                f"[bold cyan]🔄 Resuming training from checkpoint: {resume_from_checkpoint}[/]"
            )
            trainer.load_checkpoint(resume_from_checkpoint)

        metrics = trainer.train()
        final_metrics[f"stage_{s}"] = metrics
        console.print(f"[bold green]Stage {s} complete! Metrics: {metrics}[/]")

    # Save best / final model
    best_model_path = output_dir / "best_model"
    model.save_pretrained(best_model_path)
    console.print(f"[bold green]Model saved to {best_model_path}[/]")

    # Export to SentenceTransformers format
    try:
        st_export_path = output_dir / "sentence_transformers"
        model.save_as_sentence_transformer(st_export_path)
        console.print(f"[bold green]SentenceTransformers export saved to {st_export_path}[/]")
    except Exception as st_err:
        logger.debug("SentenceTransformers export failed: %s", st_err)

    # Push final model to Hugging Face Hub if requested
    if effective_push_to_hub and effective_hub_repo:
        try:
            console.print(
                f"[bold blue]🚀 Pushing final model to Hugging Face Hub: {effective_hub_repo}...[/]"
            )
            model.push_to_hub(
                repo_id=effective_hub_repo,
                private=effective_hub_private,
                commit_message="Upload CodeEmbed full-scale trained model",
            )
            console.print(
                f"[bold green]✅ Model published to https://huggingface.co/{effective_hub_repo}[/]"
            )
        except Exception as hub_err:
            logger.warning("Could not push final model to Hugging Face Hub: %s", hub_err)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    console.print("[bold green]✅ All requested training stages completed successfully.[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

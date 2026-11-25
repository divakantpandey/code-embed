"""Typed views over the YAML configs in ``configs/``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from code_embed_v3.prompts import QUERY_PROMPTS, Family

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


Direction = Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"]
HardnessMode = Literal["in_batch_negatives", "hard_negatives", "all_negatives"]


def _build[T: DataclassInstance](cls: type[T], raw: dict[str, Any], where: str) -> T:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**raw)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


@dataclass
class ModelConfig:
    name: str
    backbone: str
    pooling: Literal["mean", "cls", "lasttoken"] = "mean"
    # Only meaningful for decoder backbones (Qwen2/Qwen3); encoders are already bidirectional.
    bidirectional: bool = False
    max_seq_length: int = 1024
    # Length saved with the final model (what users and evals get). None: the backbone's position limit
    # (8192 for GTE-ModernBERT). Training runs at the shorter stage `max_seq_length` for cost.
    inference_max_seq_length: int | None = None
    matryoshka_dims: list[int] = field(default_factory=lambda: [768, 512, 256, 128, 64])
    # "auto": flash-attention 2 (needs the `flash` extra and CUDA), otherwise sdpa.
    attn_implementation: str | None = "auto"
    torch_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.matryoshka_dims != sorted(self.matryoshka_dims, reverse=True):
            raise ValueError("matryoshka_dims must be in descending order, largest = native hidden size")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        return _build(cls, load_yaml(path), str(path))


@dataclass
class SourceConfig:
    """One training source, normalised to (anchor, positive[, negative_*]) by ``data.sources``."""

    name: str
    loader: str
    family: Family
    license: str
    hf_id: str | None = None
    data_files: str | list[str] | None = None
    split: str = "train"
    max_rows: int | None = None
    # Set when the source shares provenance with an eval task, so the audit can never be skipped for it.
    overlaps_eval: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in QUERY_PROMPTS:
            raise ValueError(f"source {self.name}: unknown family {self.family!r}")


@dataclass
class DataMixConfig:
    output_dir: str
    sources: list[SourceConfig]
    seed: int = 42
    dev_rows: int = 500

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataMixConfig:
        raw = load_yaml(path)
        raw["sources"] = [
            _build(SourceConfig, s, f"{path}:sources[{i}]") for i, s in enumerate(raw["sources"])
        ]
        return _build(cls, raw, str(path))


@dataclass
class StageConfig:
    name: str
    model: str
    data_dir: str
    output_dir: str
    loss: Literal["cached_mnrl", "distill_kl"]
    init_from: str | None = None
    learning_rate: float = 1e-4
    # Global batch: split across GPUs, and with gather_across_devices every query still sees all of it.
    batch_size: int = 1024
    # GradCache chunk size. It changes memory and speed only, never the loss, so "auto" sizes it to the GPU.
    mini_batch_size: int | Literal["auto"] = "auto"
    gather_across_devices: bool = True
    # With unpadded inputs, pack GradCache chunks by token count (chunk rows x max_seq_length tokens)
    # instead of by rows, so chunks of short snippets are not left mostly empty.
    pack_mini_batches_by_tokens: bool = True
    max_steps: int = -1
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "linear"
    temperature: float = 0.05
    # CachedMNRL `directions`; ("query_to_doc", "doc_to_query") is the bidirectional GTE-style objective.
    directions: list[Direction] = field(default_factory=lambda: ["query_to_doc"])
    hardness_mode: HardnessMode | None = None
    hardness_strength: float = 0.0
    student_temperature: float | None = None
    teacher_temperature: float | None = None
    # Weight of the contrastive term when loss=distill_kl; 0 disables it.
    contrastive_weight: float = 0.0
    multi_dataset_batch_sampler: Literal["proportional", "round_robin"] = "proportional"
    bf16: bool = True
    gradient_checkpointing: bool = False
    logging_steps: int = 20
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    seed: int = 42
    report_to: list[str] = field(default_factory=lambda: ["wandb"])
    wandb_project: str = "code-embed-v3"
    # Private Hub repo that receives every checkpoint (incl. optimizer state) and the final model.
    hub_repo: str | None = None
    # None = derived from the CPUs this container may actually use.
    dataloader_num_workers: int | None = None
    dev_suite: str | None = None
    max_seq_length: int | None = None
    # Training on a source listed in `overlaps_eval` makes mteb flag the model; off unless deliberate.
    allow_eval_overlap: bool = False
    # "full": every source must be audited against all reported eval tasks (MTEB(Code), CoIR, CoREB,
    # ExecRetrieval, CoQuIR). "any": the audited flag is enough; only for dry-runs on a 2-task audit.
    audit_scope: Literal["full", "any"] = "full"
    # Loss multiplier per source name; unlisted sources weigh 1.
    source_loss_weights: dict[str, float] = field(default_factory=dict)
    # Stop and fail once the dev score falls below this fraction of the untrained model's; None disables.
    collapse_guard: float | None = None
    # Fail when fewer than this fraction of the planned optimizer steps ran; 0 disables.
    min_step_fraction: float = 0.95
    # Each batch comes from one source and partial batches are dropped, so a source smaller than one
    # batch never trains and a source of one or two batches barely does; fail instead. 0 disables.
    min_batches_per_source: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> StageConfig:
        return _build(cls, load_yaml(path), str(path))

    @property
    def scale(self) -> float:
        return 1.0 / self.temperature

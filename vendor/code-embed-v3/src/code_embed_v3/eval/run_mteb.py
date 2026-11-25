"""Official leaderboard evaluation through mteb, with honest training-data metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from code_embed_v3.data.build import read_manifest
from code_embed_v3.eval.behaviour_eval import EVAL_MAX_SEQ_LENGTH
from code_embed_v3.modeling import load_model
from code_embed_v3.prompts import mteb_model_prompts

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS = ("MTEB(Code, v1)", "CoIR", "CoREB(v1)")
TEMPLATE_MODEL = "Alibaba-NLP/gte-modernbert-base"


def training_datasets_from_manifests(data_dirs: list[str | Path]) -> set[str]:
    """Eval tasks whose data overlaps training; mteb uses this to compute the zero-shot share."""
    tasks: set[str] = set()
    for d in data_dirs:
        for meta in read_manifest(d)["sources"].values():
            tasks.update(meta.get("overlaps_eval", []))
    return tasks


class NotZeroShotError(RuntimeError):
    pass


def zero_shot_report(meta: Any, benchmarks: tuple[str, ...]) -> dict[str, int | None]:
    """mteb's own zero-shot percentage per benchmark: the number the leaderboard shows and filters on."""
    import mteb

    return {
        b: meta.zero_shot_percentage([t.metadata.name for t in mteb.get_benchmark(b).tasks])
        for b in benchmarks
    }


def require_zero_shot(report: dict[str, int | None]) -> None:
    flagged = {b: pct for b, pct in report.items() if pct is not None and pct < 100}
    if flagged:
        raise NotZeroShotError(
            f"not zero-shot: {flagged}. Remove the overlapping sources (see configs/data/eval_overlap.yaml) "
            "or pass --allow-eval-overlap for a separately reported run"
        )


def declared_meta(data_dirs: list[str | Path], base_model: str | None, template: str = TEMPLATE_MODEL) -> Any:
    """The metadata a run with this data would publish, built without loading the model.

    A registry entry serves as template only for its schema; name, training data and lineage are replaced.
    """
    import mteb

    return mteb.get_model_meta(template).model_copy(
        update={
            "name": "code-embed-v3/preflight",
            "training_datasets": training_datasets_from_manifests(data_dirs),
            "adapted_from": base_model,
        }
    )


def cast_model(model: Any, dtype: str | None) -> Any:
    """Run the model in ``dtype`` ("float16"/"bfloat16"); None keeps the checkpoint's own.

    The GTE baseline ships fp16 while fine-tunes save fp32, which made MTEB(Code) ~2x slower on the
    fine-tune; scoring both in fp16 is faster and compares like with like."""
    import torch

    if dtype is not None:
        model.to(getattr(torch, dtype))
    return model


def run(
    model_path: str | Path,
    model_name: str,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    data_dirs: list[str | Path] | None = None,
    output_folder: str | Path = "results/mteb",
    batch_size: int = 64,
    truncate_dim: int | None = None,
    base_model: str | None = None,
    tasks: list[str] | None = None,
    allow_eval_overlap: bool = False,
    max_seq_length: int | None = EVAL_MAX_SEQ_LENGTH,
    dtype: str | None = None,
) -> Any:
    import mteb
    from mteb.cache import ResultCache
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

    model = cast_model(load_model(model_path, truncate_dim=truncate_dim), dtype)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    has_prompts = "nl2code" in getattr(model, "prompts", {})
    wrapper = SentenceTransformerEncoderWrapper(
        model, model_prompts=mteb_model_prompts() if has_prompts else None
    )
    wrapper.mteb_model_meta = wrapper.mteb_model_meta.model_copy(
        update={
            "name": model_name,
            # mteb adds the base model's declared data itself through `adapted_from`.
            "training_datasets": training_datasets_from_manifests(data_dirs or []),
            "adapted_from": base_model,
            "use_instructions": has_prompts,
            "license": "apache-2.0",
            "open_weights": True,
        }
    )

    report = zero_shot_report(wrapper.mteb_model_meta, benchmarks)
    logger.info("zero-shot %%: %s", report)
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_shot.json").write_text(json.dumps(report, indent=2))
    if not allow_eval_overlap:
        require_zero_shot(report)

    selected: list[Any] = []
    if tasks:
        selected = [mteb.get_task(t) for t in tasks]
    else:
        seen: set[str] = set()
        for name in benchmarks:
            for task in mteb.get_benchmark(name).tasks:
                if task.metadata.name not in seen:
                    seen.add(task.metadata.name)
                    selected.append(task)
    logger.info("evaluating %s on %d tasks", model_name, len(selected))
    return mteb.evaluate(
        wrapper, selected, encode_kwargs={"batch_size": batch_size}, cache=ResultCache(str(output_folder))
    )

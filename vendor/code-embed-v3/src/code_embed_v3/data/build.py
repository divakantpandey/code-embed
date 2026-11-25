"""Materialise a data mix to disk with a manifest that later stages read."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_from_disk

from code_embed_v3.config import DataMixConfig
from code_embed_v3.data.sources import build_source
from code_embed_v3.prompts import QUERY_PROMPTS, Family

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
DEV_DIR = "dev"


def build_mix(cfg: DataMixConfig, only: list[str] | None = None) -> Path:
    """Build each source; the first ``dev_rows`` shuffled rows are held out for the dev suite."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(out) if (out / MANIFEST).exists() else {"sources": {}}
    for src in cfg.sources:
        if only and src.name not in only:
            continue
        logger.info("building %s (%s, %s)", src.name, src.loader, src.family)
        ds = build_source(src).shuffle(seed=cfg.seed)
        n_dev = min(cfg.dev_rows, len(ds) // 10)
        ds.select(range(n_dev)).save_to_disk(str(out / DEV_DIR / src.name))
        ds = ds.select(range(n_dev, len(ds)))
        ds.save_to_disk(str(out / src.name))
        manifest["sources"][src.name] = {**asdict(src), "rows": len(ds), "dev_rows": n_dev, "audited": False}
        logger.info("  %s train rows, %s dev rows", f"{len(ds):,}", n_dev)
    write_manifest(out, manifest)
    return out


def source_dir(data_dir: str | Path, name: str, meta: dict[str, Any]) -> Path:
    """A source's data: ``meta["path"]`` when a combined mix points at another mix, else ``data_dir/name``."""
    return Path(meta["path"]) if meta.get("path") else Path(data_dir) / name


def read_manifest(data_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / MANIFEST).read_text())


def write_manifest(data_dir: str | Path, manifest: dict[str, Any]) -> None:
    (Path(data_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2))


def load_mix(
    data_dir: str | Path,
    require_audit: bool = True,
    allow_eval_overlap: bool = False,
    audited_against: Collection[str] = (),
) -> tuple[DatasetDict, dict[str, Family]]:
    """Load every source of a built mix; returns the datasets and each source's task family.

    ``audited_against``: tasks every source must have been audited against. A dry-run audit covers two
    tasks only, so the `audited` flag alone would let such a mix train a model that is then scored on
    the benchmarks it was never checked against."""
    data_dir = Path(data_dir)
    manifest = read_manifest(data_dir)
    unaudited = [n for n, m in manifest["sources"].items() if not m.get("audited")]
    if require_audit and unaudited:
        raise RuntimeError(f"run `ce3 audit` first; unaudited sources: {unaudited}")
    missing = {
        n: sorted(set(audited_against) - set(m.get("audited_against", [])))
        for n, m in manifest["sources"].items()
    }
    missing = {n: tasks for n, tasks in missing.items() if tasks}
    if require_audit and missing:
        shown = {n: t[:5] + (["..."] if len(t) > 5 else []) for n, t in missing.items()}
        raise RuntimeError(f"run the full `ce3 audit`; sources not audited against every eval task: {shown}")
    # mteb flags a model by provenance, not by rows: training on these makes it non-zero-shot on the
    # listed tasks no matter how thoroughly the audit removed their test rows.
    overlapping = {n: m["overlaps_eval"] for n, m in manifest["sources"].items() if m.get("overlaps_eval")}
    if overlapping and not allow_eval_overlap:
        raise RuntimeError(
            "sources share provenance with eval tasks and would make the model non-zero-shot: "
            f"{overlapping}; set allow_eval_overlap: true in the stage config only for a "
            "separately reported run"
        )
    datasets: dict[str, Dataset] = {}
    families: dict[str, Family] = {}
    for name, meta in manifest["sources"].items():
        path = source_dir(data_dir, name, meta)
        ds = load_from_disk(str(path))
        if not isinstance(ds, Dataset):
            raise TypeError(f"{path} is not a single Dataset")
        if meta["family"] not in QUERY_PROMPTS:
            raise ValueError(f"{name}: unknown family {meta['family']!r} in manifest")
        datasets[name] = ds
        families[name] = cast(Family, meta["family"])
    return DatasetDict(datasets), families

"""Loaders that normalise each training source to ``anchor`` / ``positive`` [/ ``negative_i``] columns.

Every loader was written against the source's actual schema on the Hub (checked 2026-10-29); if a
source changes upstream the loader fails loudly on the missing column instead of training on junk.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from collections.abc import Callable, Iterator
from typing import Any

from datasets import Dataset, load_dataset

from code_embed_v3.config import SourceConfig

Row = dict[str, str]
Loader = Callable[[SourceConfig], Iterator[Row]]

LOADERS: dict[str, Loader] = {}


def register(name: str) -> Callable[[Loader], Loader]:
    def deco(fn: Loader) -> Loader:
        LOADERS[name] = fn
        return fn

    return deco


def _load(src: SourceConfig) -> Any:
    if src.data_files is not None:
        files = [src.data_files] if isinstance(src.data_files, str) else src.data_files
        builder = "parquet" if all(f.endswith(".parquet") for f in files) else "json"
        return load_dataset(builder, data_files=src.data_files, split="train", streaming=True)
    if src.hf_id is None:
        raise ValueError(f"source {src.name}: hf_id or data_files is required")
    return load_dataset(src.hf_id, split=src.split, streaming=True, **src.options.get("load_kwargs", {}))


@register("columns")
def load_columns(src: SourceConfig) -> Iterator[Row]:
    """Plain pair sources; ``options.anchor`` / ``options.positive`` name the columns."""
    a, p = src.options["anchor"], src.options["positive"]
    for row in _load(src):
        yield {"anchor": row[a], "positive": row[p]}


@register("cornstack")
def load_cornstack(src: SourceConfig) -> Iterator[Row]:
    """CoRNStack ships consistency-filtered negatives; keep ``options.num_negatives`` of them.

    ``options.reverse`` yields code -> description pairs (code2nl). Its negatives are code, not
    descriptions, so they are never used in that direction.
    """
    reverse = bool(src.options.get("reverse", False))
    k = 0 if reverse else int(src.options.get("num_negatives", 0))
    for row in _load(src):
        if reverse:
            yield {"anchor": row["document"], "positive": row["query"]}
            continue
        out = {"anchor": row["query"], "positive": row["document"]}
        if k:
            negs = row["negatives"][:k]
            if len(negs) < k:
                continue
            out.update({f"negative_{i + 1}": n for i, n in enumerate(negs)})
        yield out


@register("commitpackft")
def load_commitpackft(src: SourceConfig) -> Iterator[Row]:
    """Commit message -> unified diff, the same shape as MTEB's CodeEditSearchRetrieval."""
    context = int(src.options.get("diff_context", 3))
    for row in _load(src):
        diff = "".join(
            difflib.unified_diff(
                row["old_contents"].splitlines(keepends=True),
                row["new_contents"].splitlines(keepends=True),
                fromfile=row["old_file"],
                tofile=row["new_file"],
                n=context,
            )
        )
        if diff:
            yield {"anchor": row["message"], "positive": diff}


@register("apps")
def load_apps(src: SourceConfig) -> Iterator[Row]:
    """Problem statement -> one reference solution (``solutions`` is a JSON-encoded list)."""
    for row in _load(src):
        try:
            solutions = json.loads(row["solutions"]) if row["solutions"] else []
        except json.JSONDecodeError:
            continue
        if solutions:
            yield {"anchor": row["question"], "positive": solutions[0]}


# deepmind/code_contests `solutions.language` ClassLabel ids.
_CC_LANGS = {1: "python", 2: "cpp", 3: "python", 4: "java"}


@register("code_contests_xlang")
def load_code_contests_xlang(src: SourceConfig) -> Iterator[Row]:
    """Accepted solutions to the same problem in two different languages form a code2code pair."""
    rng = random.Random(src.options.get("seed", 0))
    per_problem = int(src.options.get("pairs_per_problem", 2))
    for row in _load(src):
        by_lang: dict[str, list[str]] = {}
        for lang_id, code in zip(row["solutions"]["language"], row["solutions"]["solution"], strict=True):
            if lang_id in _CC_LANGS:
                by_lang.setdefault(_CC_LANGS[lang_id], []).append(code)
        langs = sorted(by_lang)
        if len(langs) < 2:
            continue
        for _ in range(per_problem):
            la, lb = rng.sample(langs, 2)
            yield {"anchor": rng.choice(by_lang[la]), "positive": rng.choice(by_lang[lb])}


_HUNK = re.compile(r"@@ -(\S+) \+(\S+) @@")


def reverse_patch(patch: str) -> str:
    """Invert a unified diff, keeping each change block in ``-`` then ``+`` order."""
    out: list[str] = []
    removed: list[str] = []
    added: list[str] = []
    old_path = ""
    # Inside a hunk a removed "-- x" reads as "--- x", so file headers only count between hunks.
    in_hunk = False

    def flush() -> None:
        out.extend(removed + added)
        removed.clear()
        added.clear()

    for line in patch.splitlines():
        if line.startswith("diff "):
            flush()
            in_hunk = False
            out.append(line)
        elif not in_hunk and line.startswith("--- "):
            old_path = line[4:]
        elif not in_hunk and line.startswith("+++ "):
            out.extend(["--- " + line[4:], "+++ " + old_path])
        elif line.startswith("@@ "):
            flush()
            in_hunk = True
            out.append(_HUNK.sub(lambda m: f"@@ -{m[2]} +{m[1]} @@", line, count=1))
        elif line.startswith("+"):
            removed.append("-" + line[1:])
        elif line.startswith("-"):
            added.append("+" + line[1:])
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out) + "\n"


@register("swe_smith")
def load_swe_smith(src: SourceConfig) -> Iterator[Row]:
    """Issue -> fixing patch. SWE-smith's ``patch`` injects the bug, so it is reversed into the fix."""
    for row in _load(src):
        stmt = (row["problem_statement"] or "").strip()
        if stmt and row["patch"]:
            yield {"anchor": stmt, "positive": reverse_patch(row["patch"])}


@register("beir")
def load_beir(src: SourceConfig) -> Iterator[Row]:
    """BEIR-layout sources (``default`` qrels + ``corpus`` + ``queries`` configs), e.g. CoIR train splits.

    These are in-domain for the matching eval task: list it in ``overlaps_eval`` so the model's
    mteb ``training_datasets`` metadata and the zero-shot percentage stay honest.
    """
    if src.hf_id is None:
        raise ValueError(f"source {src.name}: beir loader needs hf_id")
    corpus = {r["_id"]: r["text"] for r in load_dataset(src.hf_id, "corpus", split="corpus")}
    queries = {r["_id"]: r["text"] for r in load_dataset(src.hf_id, "queries", split="queries")}
    for qrel in load_dataset(src.hf_id, "default", split=src.split):
        if qrel["score"] <= 0:
            continue
        q, d = queries.get(qrel["query-id"]), corpus.get(qrel["corpus-id"])
        if q and d:
            yield {"anchor": q, "positive": d}


def _keep(row: Row, min_chars: int, max_chars: int) -> bool:
    return all(isinstance(v, str) and min_chars <= len(v.strip()) <= max_chars for v in row.values())


def build_source(src: SourceConfig, min_chars: int = 8, max_chars: int = 16_000) -> Dataset:
    """Materialise one source, dropping empty/oversized rows and exact duplicate pairs."""
    if src.loader not in LOADERS:
        raise KeyError(f"source {src.name}: unknown loader {src.loader!r}; known: {sorted(LOADERS)}")
    seen: set[int] = set()

    def rows() -> Iterator[Row]:
        n = 0
        for row in LOADERS[src.loader](src):
            if not _keep(row, min_chars, max_chars):
                continue
            key = hash((row["anchor"], row["positive"]))
            if key in seen:
                continue
            seen.add(key)
            yield row
            n += 1
            if src.max_rows is not None and n >= src.max_rows:
                return

    return Dataset.from_generator(rows)

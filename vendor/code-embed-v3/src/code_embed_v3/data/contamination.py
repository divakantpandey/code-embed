"""Train/test contamination audit against the mteb benchmarks we report on.

Only test *queries* and the documents *relevant to them* are indexed: several CoIR corpora (e.g.
CosQA) also contain train-partition documents, and indexing the whole corpus would flag legitimate
training data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset, List, Value
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def text_of(value: Any) -> str:
    """mteb queries are usually str, but multi-turn tasks hold lists of turns."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(text_of(v) for k, v in value.items() if k in ("title", "text", "content"))
    if isinstance(value, Iterable):
        return " ".join(text_of(v) for v in value)
    return str(value)


def exact_key(text: str) -> str:
    return hashlib.blake2b(normalise(text).encode(), digest_size=16).hexdigest()


def minhash(text: str, num_perm: int, ngram: int) -> MinHash | None:
    tokens = _TOKEN.findall(normalise(text))
    if len(tokens) < ngram:
        return None
    mh = MinHash(num_perm=num_perm)
    mh.update_batch([" ".join(tokens[i : i + ngram]).encode() for i in range(len(tokens) - ngram + 1)])
    return mh


PARALLEL_CHUNK = 256


class EvalIndex:
    """Exact + MinHash-LSH index over eval texts, tagged with the task each text came from."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, ngram: int = 5) -> None:
        self.threshold, self.num_perm, self.ngram = threshold, num_perm, ngram
        self.exact: dict[str, str] = {}
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._n = 0

    def add(self, task: str, text: str) -> None:
        self.add_many(task, [text])

    def add_many(self, task: str, texts: Sequence[str], num_proc: int = 1) -> None:
        """Same index as ``add`` per text; MinHash, the costly part, runs on ``num_proc`` processes."""
        for text in texts:
            self.exact.setdefault(exact_key(text), task)
        sign = partial(minhash, num_perm=self.num_perm, ngram=self.ngram)
        if num_proc > 1 and len(texts) >= 4 * PARALLEL_CHUNK:
            with ProcessPoolExecutor(num_proc) as pool:
                hashes = list(pool.map(sign, texts, chunksize=PARALLEL_CHUNK))
        else:
            hashes = [sign(t) for t in texts]
        for mh in hashes:
            if mh is not None:
                self.lsh.insert(f"{task}::{self._n}", mh)
                self._n += 1

    def __len__(self) -> int:
        return len(self.exact)

    def hits(self, text: str) -> set[str]:
        """Eval tasks this text exactly or near-duplicates."""
        found = set()
        if (task := self.exact.get(exact_key(text))) is not None:
            found.add(task)
        mh = minhash(text, self.num_perm, self.ngram)
        if mh is not None:
            found.update(key.split("::", 1)[0] for key in self.lsh.query(mh))
        return found


def index_mteb_tasks(task_names: list[str], index: EvalIndex | None = None, num_proc: int = 1) -> EvalIndex:
    import mteb

    index = index or EvalIndex()
    for name in task_names:
        task = mteb.get_task(name)
        task.load_data()
        if hasattr(task, "convert_v1_dataset_format_to_v2"):
            task.convert_v1_dataset_format_to_v2(num_proc=None)
        if task.dataset is None:
            raise RuntimeError(f"{name}: mteb loaded no data")
        for subset in task.dataset.values():
            for split_name in task.metadata.eval_splits:
                split = subset.get(split_name)
                if not split or "queries" not in split:
                    logger.warning("%s/%s has no retrieval layout; skipped", name, split_name)
                    continue
                relevant = split.get("relevant_docs") or {}
                wanted_docs = {d for docs in relevant.values() for d in docs}
                texts = [text_of(q["text"]) for q in split["queries"]]
                texts += [text_of(d.get("text", "")) for d in split["corpus"] if d["id"] in wanted_docs]
                index.add_many(name, texts, num_proc)
        logger.info("indexed %s (%d unique eval texts so far)", name, len(index))
    return index


def index_texts(texts: dict[str, list[str]], index: EvalIndex, num_proc: int = 1) -> EvalIndex:
    """Add non-mteb held-out benchmarks (ExecRetrieval, CoQuIR) under their own task names."""
    for name, items in texts.items():
        index.add_many(name, items, num_proc)
    return index


# The index of a running audit, inherited by forked map workers. Capturing it in the mapped function
# instead makes `datasets` pickle the whole index (~360k MinHashes) once per worker and per call: minutes
# of single-core serialising and a full copy in every worker's RAM.
_AUDIT_INDEX: EvalIndex | None = None


def _flag(row: dict[str, Any], text_cols: list[str]) -> dict[str, Any]:
    if _AUDIT_INDEX is None:
        raise RuntimeError("audit index missing in this worker; workers must be forked")
    tasks: set[str] = set()
    for c in text_cols:
        tasks |= _AUDIT_INDEX.hits(row[c])
    return {"_hits": sorted(tasks)}


def audit_dataset(ds: Dataset, index: EvalIndex, num_proc: int = 1) -> tuple[Dataset, Counter[str]]:
    """Drop every row whose any text column hits the eval index; return kept rows and hits per task."""
    import multiprocess

    global _AUDIT_INDEX
    text_cols = [c for c in ds.column_names if c == "anchor" or c == "positive" or c.startswith("negative_")]
    if num_proc > 1 and multiprocess.get_start_method() != "fork":
        num_proc = 1
    # Declared, not inferred: a writer batch without any hit would type `_hits` as list<null> and the
    # next batch with a hit would fail to cast.
    features = ds.features.copy()
    features["_hits"] = List(Value("string"))
    _AUDIT_INDEX = index
    try:
        flagged = ds.map(
            _flag,
            fn_kwargs={"text_cols": text_cols},
            num_proc=num_proc,
            features=features,
            desc="contamination audit",
        )
    finally:
        _AUDIT_INDEX = None
    counts: Counter[str] = Counter(t for hits in flagged["_hits"] for t in hits)
    kept = flagged.filter(lambda r: not r["_hits"], num_proc=num_proc).remove_columns("_hits")
    return kept, counts


def full_audit_tasks() -> list[str]:
    """Every task a training source must be audited against before a real (non-dry-run) stage trains on it:
    all of MTEB(Code, v1), CoIR and CoREB(v1), plus ExecRetrieval and each CoQuIR subset."""
    import mteb

    from code_embed_v3.eval.behaviour_eval import COQUIR_SUBSETS
    from code_embed_v3.eval.run_mteb import DEFAULT_BENCHMARKS

    tasks = {t.metadata.name for b in DEFAULT_BENCHMARKS for t in mteb.get_benchmark(b).tasks}
    return sorted(tasks | {"ExecRetrieval"} | {f"CoQuIR-{s}" for s in COQUIR_SUBSETS})


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True))

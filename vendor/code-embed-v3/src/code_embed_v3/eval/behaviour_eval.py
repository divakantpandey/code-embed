"""Correctness-aware retrieval: does the model rank code that works above code that only looks right?

- ExecRetrieval (AaryanK/ExecRetrieval, CC BY 4.0, EMNLP 2026): 939 task descriptions; each has an
  execution-verified implementation and up to four single-edit buggy copies in a 4,694-snippet
  corpus. ``exec@1``: the top hit is a correct implementation.
- CoQuIR (jiahuimbzuai/CoQuIR, Apache-2.0): queries with better and worse code (buggy vs fixed,
  slow vs efficient, vulnerable vs safe). ``pairwise``: share of (good, bad) pairs ranked the right way.

Both are held-out: training rows are audited against them, and nothing is tuned on them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

EXEC_REPO = "AaryanK/ExecRetrieval"
COQUIR_REPO = "jiahuimbzuai/CoQuIR"
# GTE-ModernBERT's native context; every model is scored at this length.
EVAL_MAX_SEQ_LENGTH = 8192
COQUIR_SUBSETS = ("CodeNet-B", "CodeNet-E", "Defects4J", "CVEFixes", "SafeCoder", "DepreAPI", "SQLR2")


def _jsonl(repo: str, filename: str) -> list[dict[str, Any]]:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_exec_retrieval() -> tuple[list[dict[str, Any]], dict[str, str]]:
    queries = _jsonl(EXEC_REPO, "data/queries.jsonl")
    corpus = {str(r["corpus_id"]): r["code"] for r in _jsonl(EXEC_REPO, "data/corpus.jsonl")}
    return queries, corpus


def load_coquir(subset: str) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    # Field names as shipped on the Hub (checked 2026-10-29): `query-id`, `doc-id`. A few rows ship
    # `"text": NaN` (Defects4J, SQLR2 negatives); scoring already skips qrel ids missing from the corpus.
    queries = {
        str(r["query-id"]): r["text"]
        for r in _jsonl(COQUIR_REPO, f"{subset}/query.jsonl")
        if isinstance(r["text"], str)
    }
    corpus = {
        str(r["doc-id"]): r["text"]
        for r in _jsonl(COQUIR_REPO, f"{subset}/corpus.jsonl")
        if isinstance(r["text"], str)
    }
    qrels = _jsonl(COQUIR_REPO, f"{subset}/qrels.jsonl")
    return queries, corpus, qrels


def external_texts() -> dict[str, list[str]]:
    """Every query and document of both benchmarks, for the contamination audit."""
    queries, corpus = load_exec_retrieval()
    texts = {"ExecRetrieval": [q["query"] for q in queries] + list(corpus.values())}
    for subset in COQUIR_SUBSETS:
        q, c, _ = load_coquir(subset)
        texts[f"CoQuIR-{subset}"] = list(q.values()) + list(c.values())
    return texts


def _encode(
    model: Any, texts: list[str], query: bool, batch_size: int, prompt_name: str = "nl2code"
) -> np.ndarray:
    # Models without our prompts (e.g. the GTE-ModernBERT baseline) encode queries plain.
    kwargs = {"prompt_name": prompt_name} if query and prompt_name in getattr(model, "prompts", {}) else {}
    # float32 always: the GTE baseline ships fp16 and encodes to float16, whose numpy matmul has no BLAS
    # kernel (a 26k x 53k CoQuIR product ran for over an hour) and whose rounding ties near-twin scores.
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, **kwargs), dtype=np.float32
    )


def exec_retrieval_outcomes(
    sims: np.ndarray, corpus_ids: list[str], queries: list[dict[str, Any]]
) -> dict[str, list[int]]:
    """Per query, in query order: 1 if the query counts towards each metric."""
    order = np.argsort(-sims, axis=1)
    idx = {cid: j for j, cid in enumerate(corpus_ids)}
    out: dict[str, list[int]] = {"exec@1": [], "exec@10": [], "beats_paired_distractors": []}
    for qi, q in enumerate(queries):
        correct = {str(c) for c in q["correct_corpus_ids"]}
        ranked = [corpus_ids[j] for j in order[qi, :10]]
        best_correct = max(sims[qi, idx[c]] for c in correct if c in idx)
        paired = [sims[qi, idx[str(d)]] for d in q.get("paired_distractor_corpus_ids") or [] if str(d) in idx]
        out["exec@1"].append(int(ranked[0] in correct))
        out["exec@10"].append(int(bool(correct & set(ranked))))
        out["beats_paired_distractors"].append(int(all(best_correct > s for s in paired)))
    return out


def exec_retrieval_scores(
    sims: np.ndarray, corpus_ids: list[str], queries: list[dict[str, Any]]
) -> dict[str, float]:
    outcomes = exec_retrieval_outcomes(sims, corpus_ids, queries)
    return {k: sum(v) / len(v) for k, v in outcomes.items()}


def coquir_outcomes(
    sims: np.ndarray, query_ids: list[str], corpus_ids: list[str], qrels: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    """Per scored qrel row: its qid, pairs ranked right, pairs, and ndcg@10 (None without positives).
    Pairs of one query share its embedding, so significance tests resample queries, not pairs."""
    q_idx = {q: i for i, q in enumerate(query_ids)}
    d_idx = {d: j for j, d in enumerate(corpus_ids)}
    out: dict[str, list[Any]] = {"qid": [], "pair_hits": [], "pairs": [], "ndcg@10": []}
    for rel in qrels:
        qi = q_idx.get(str(rel["qid"]))
        if qi is None:
            continue
        pos = [d_idx[str(d)] for d in rel.get("pos-docids") or [] if str(d) in d_idx]
        neg = [d_idx[str(d)] for d in rel.get("neg-docids") or [] if str(d) in d_idx]
        hits = sum(int(sims[qi, p] > sims[qi, n]) for p in pos for n in neg)
        ndcg = None
        if pos:
            top = np.argsort(-sims[qi])[:10]
            gains = [1.0 / np.log2(r + 2) for r, j in enumerate(top) if j in set(pos)]
            ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos), 10)))
            ndcg = float(sum(gains) / ideal)
        out["qid"].append(str(rel["qid"]))
        out["pair_hits"].append(hits)
        out["pairs"].append(len(pos) * len(neg))
        out["ndcg@10"].append(ndcg)
    return out


def coquir_scores(
    sims: np.ndarray, query_ids: list[str], corpus_ids: list[str], qrels: list[dict[str, Any]]
) -> dict[str, float]:
    return coquir_summary(coquir_outcomes(sims, query_ids, corpus_ids, qrels))


def coquir_summary(outcomes: dict[str, list[Any]]) -> dict[str, float]:
    pairs = sum(outcomes["pairs"])
    ndcg = [x for x in outcomes["ndcg@10"] if x is not None]
    return {
        "pairwise": sum(outcomes["pair_hits"]) / pairs if pairs else float("nan"),
        "ndcg@10": float(np.mean(ndcg)) if ndcg else float("nan"),
    }


def evaluate(
    model: Any,
    batch_size: int = 64,
    subsets: tuple[str, ...] = COQUIR_SUBSETS,
    max_seq_length: int | None = EVAL_MAX_SEQ_LENGTH,
) -> dict[str, Any]:
    """``max_seq_length`` is applied to every model so base and candidate see the same text; a
    checkpoint's own (training) length otherwise cuts long code: 86/465 Defects4J pairs are identical
    at 256 tokens. ``per_query`` holds each benchmark's per-query outcomes for paired significance tests
    (``significance.compare``)."""
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    queries, corpus = load_exec_retrieval()
    corpus_ids = list(corpus)
    q = _encode(model, [x["query"] for x in queries], True, batch_size)
    d = _encode(model, [corpus[c] for c in corpus_ids], False, batch_size)
    exec_outcomes = exec_retrieval_outcomes(q @ d.T, corpus_ids, queries)
    results: dict[str, Any] = {
        "max_seq_length": model.max_seq_length,
        "ExecRetrieval": {k: sum(v) / len(v) for k, v in exec_outcomes.items()},
    }
    per_query: dict[str, Any] = {"ExecRetrieval": exec_outcomes}
    for subset in subsets:
        qs, cs, qrels = load_coquir(subset)
        qids, cids = list(qs), list(cs)
        prompt = "text2sql" if subset == "SQLR2" else "nl2code"
        sims = (
            _encode(model, [qs[i] for i in qids], True, batch_size, prompt_name=prompt)
            @ _encode(model, [cs[i] for i in cids], False, batch_size, prompt_name=prompt).T
        )
        per_query[f"CoQuIR-{subset}"] = coquir_outcomes(sims, qids, cids, qrels)
        results[f"CoQuIR-{subset}"] = coquir_summary(per_query[f"CoQuIR-{subset}"])
    logger.info("behaviour eval: %s", results)
    return {**results, "per_query": per_query}


def mteb_code_mean(results_dir: Path) -> float:
    import mteb

    from code_embed_v3.release import mteb_scores

    tasks = {t.metadata.name for t in mteb.get_benchmark("MTEB(Code, v1)").tasks}
    scores = {k: v for k, v in mteb_scores(results_dir).items() if k in tasks}
    if len(scores) != len(tasks):
        raise ValueError(f"{results_dir}: {len(scores)}/{len(tasks)} MTEB(Code, v1) tasks found")
    return sum(scores.values()) / len(scores)


def no_collapse(
    baseline: dict[str, Any], candidate: dict[str, Any], min_kept: float = 0.5
) -> tuple[bool, dict[str, float]]:
    """Fail when exec@10 or any CoQuIR ndcg@10 falls below ``min_kept`` of the baseline.

    A collapsed model scores near random (exec@10 0.967 -> 0.003 in the first dry-run) while still
    passing the zero-shot gate, so the dry-run needs a check that only looks for that, not for gains.
    """
    pairs = {
        "ExecRetrieval exec@10": (baseline["ExecRetrieval"]["exec@10"], candidate["ExecRetrieval"]["exec@10"])
    }
    for k in baseline:
        if k.startswith("CoQuIR-") and k in candidate and np.isfinite(baseline[k]["ndcg@10"]):
            pairs[f"{k} ndcg@10"] = (baseline[k]["ndcg@10"], candidate[k]["ndcg@10"])
    kept = {name: cand / base if base > 0 else 1.0 for name, (base, cand) in pairs.items()}
    return all(v >= min_kept for v in kept.values()), kept


def pilot_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    mteb_baseline: float,
    mteb_candidate: float,
    min_exec_rel_gain: float = 0.10,
    max_mteb_drop: float = 0.005,
    max_coquir_drop: float = 0.05,
) -> tuple[bool, dict[str, float]]:
    """Continue only if exec@1 rises by ``min_exec_rel_gain`` relative to the baseline (0.10 = +10%),
    MTEB(Code) drops by at most ``max_mteb_drop`` (0.005 = half a point on mteb's 0-1 scale) and mean
    CoQuIR pairwise accuracy by at most ``max_coquir_drop``. CoQuIR's subsets (Defects4J, CVEFixes, ...)
    are the out-of-distribution check: training mutants can mirror ExecRetrieval's bug types, so its
    gain alone could be in-distribution. Every fine-tune so far loses ~0.1 Defects4J pairwise, hence the
    looser default; per-subset drops are reported so that cost stays visible."""
    base_exec = baseline["ExecRetrieval"]["exec@1"]
    gain = candidate["ExecRetrieval"]["exec@1"] - base_exec
    rel_gain = gain / base_exec
    drop = mteb_baseline - mteb_candidate
    # Subsets scored in both runs only; a subset with no pairs is NaN.
    subsets = [
        k
        for k in baseline
        if k.startswith("CoQuIR-")
        and k in candidate
        and np.isfinite(baseline[k]["pairwise"])
        and np.isfinite(candidate[k]["pairwise"])
    ]
    if not subsets:
        raise ValueError("no CoQuIR subset scored in both baseline and candidate")
    coquir_base = float(np.mean([baseline[k]["pairwise"] for k in subsets]))
    coquir_cand = float(np.mean([candidate[k]["pairwise"] for k in subsets]))
    coquir_drop = coquir_base - coquir_cand
    report = {
        "exec@1_gain": gain,
        "exec@1_rel_gain": rel_gain,
        "mteb_code_drop": drop,
        "mteb_code_baseline": mteb_baseline,
        "mteb_code_candidate": mteb_candidate,
        "coquir_pairwise_drop": coquir_drop,
        "coquir_pairwise_baseline": coquir_base,
        "coquir_pairwise_candidate": coquir_cand,
        **{f"{k}_pairwise_drop": baseline[k]["pairwise"] - candidate[k]["pairwise"] for k in subsets},
    }
    return rel_gain >= min_exec_rel_gain and drop <= max_mteb_drop and coquir_drop <= max_coquir_drop, report

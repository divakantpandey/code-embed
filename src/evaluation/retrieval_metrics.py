"""Retrieval metrics for code-search evaluation.

Implements standard information-retrieval metrics operating on numpy arrays
for efficient batch computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual retrieval metrics
# ---------------------------------------------------------------------------


def mrr_at_k(rankings: list[list[int]], k: int) -> float:
    """Compute Mean Reciprocal Rank at *k*.

    Parameters
    ----------
    rankings:
        For each query, a ranked list of document indices (best first).
        The *relevant* document is assumed to be the one with the same
        positional index as the query (i.e. ``rankings[i]`` should ideally
        contain ``i`` near the top).
    k:
        Cutoff depth.

    Returns
    -------
    float
        MRR@k averaged over all queries, in [0, 1].
    """
    reciprocal_ranks: list[float] = []
    for query_idx, ranked_docs in enumerate(rankings):
        truncated = ranked_docs[:k]
        try:
            rank = truncated.index(query_idx) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def recall_at_k(rankings: list[list[int]], k: int) -> float:
    """Compute Recall at *k* (binary relevance, one relevant doc per query).

    Parameters
    ----------
    rankings:
        For each query, a ranked list of document indices (best first).
    k:
        Cutoff depth.

    Returns
    -------
    float
        Fraction of queries for which the relevant document appears in the
        top-*k* results.
    """
    hits = 0
    for query_idx, ranked_docs in enumerate(rankings):
        if query_idx in ranked_docs[:k]:
            hits += 1
    return hits / len(rankings) if rankings else 0.0


def ndcg_at_k(
    rankings: list[list[int]],
    relevance: list[list[float]],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at *k*.

    Parameters
    ----------
    rankings:
        For each query, a ranked list of document indices.
    relevance:
        ``relevance[q][d]`` gives the graded relevance of document *d* for
        query *q*.  If not all documents are scored, missing entries should
        be 0.
    k:
        Cutoff depth.

    Returns
    -------
    float
        NDCG@k averaged over all queries.
    """
    ndcg_scores: list[float] = []
    for q_idx, ranked_docs in enumerate(rankings):
        # Gather relevance values for top-k ranked documents
        gains = []
        for doc_idx in ranked_docs[:k]:
            if doc_idx < len(relevance[q_idx]):
                gains.append(relevance[q_idx][doc_idx])
            else:
                gains.append(0.0)
        gains_arr = np.asarray(gains, dtype=np.float64)

        # DCG
        discounts = np.log2(np.arange(2, len(gains_arr) + 2, dtype=np.float64))
        dcg = float(np.sum(gains_arr / discounts))

        # Ideal DCG (sort all relevance values descending)
        ideal_gains = np.sort(np.asarray(relevance[q_idx], dtype=np.float64))[::-1][:k]
        ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2, dtype=np.float64))
        idcg = float(np.sum(ideal_gains / ideal_discounts))

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def map_at_k(rankings: list[list[int]], k: int) -> float:
    """Compute Mean Average Precision at *k* (binary relevance).

    Each query is assumed to have exactly one relevant document (the one
    sharing the same positional index as the query).

    Parameters
    ----------
    rankings:
        For each query, a ranked list of document indices (best first).
    k:
        Cutoff depth.

    Returns
    -------
    float
        MAP@k averaged over all queries.
    """
    avg_precisions: list[float] = []
    for query_idx, ranked_docs in enumerate(rankings):
        truncated = ranked_docs[:k]
        try:
            rank = truncated.index(query_idx) + 1  # 1-indexed
            # With binary relevance and a single relevant doc, AP = 1/rank
            avg_precisions.append(1.0 / rank)
        except ValueError:
            avg_precisions.append(0.0)
    return float(np.mean(avg_precisions)) if avg_precisions else 0.0


# ---------------------------------------------------------------------------
# Convenience: compute all retrieval metrics at once
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    query_embeddings: NDArray[np.float32],
    doc_embeddings: NDArray[np.float32],
    relevance_labels: NDArray[np.float32] | None = None,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Compute a full suite of retrieval metrics.

    When ``relevance_labels`` is *None*, binary relevance is assumed: each
    query *i* has a single relevant document *i* (paired evaluation).

    Parameters
    ----------
    query_embeddings:
        Shape ``(n_queries, dim)``.
    doc_embeddings:
        Shape ``(n_docs, dim)``.
    relevance_labels:
        Optional graded relevance matrix of shape ``(n_queries, n_docs)``.
        If *None*, diagonal (paired) binary relevance is used.
    ks:
        Cutoff values to evaluate.  Defaults to ``[1, 5, 10, 100]``.

    Returns
    -------
    dict[str, float]
        Mapping of metric names (e.g. ``"MRR@10"``) to values.
    """
    if ks is None:
        ks = [1, 5, 10, 100]

    n_queries = query_embeddings.shape[0]
    n_docs = doc_embeddings.shape[0]

    logger.info("Computing retrieval metrics for %d queries × %d docs", n_queries, n_docs)

    # --- Normalise embeddings and compute cosine similarity matrix ----------
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)
    d_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)
    # (n_queries, n_docs)
    sim_matrix: NDArray[np.float32] = q_norm @ d_norm.T

    # --- Derive rankings (descending similarity) ---------------------------
    # For large matrices, argpartition + sort is faster than full argsort.
    max_k = min(max(ks), n_docs)
    rankings: list[list[int]] = []
    for q in range(n_queries):
        sims = sim_matrix[q]
        if max_k < n_docs:
            # Partial sort: get top-max_k indices, then sort them properly
            top_indices = np.argpartition(sims, -max_k)[-max_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        else:
            top_indices = np.argsort(sims)[::-1]
        rankings.append(top_indices.tolist())

    # --- Build relevance matrix if not provided ----------------------------
    if relevance_labels is None:
        # Diagonal binary relevance (paired assumption)
        relevance: list[list[float]] = []
        for q in range(n_queries):
            row = [0.0] * n_docs
            if q < n_docs:
                row[q] = 1.0
            relevance.append(row)
    else:
        relevance = relevance_labels.tolist()

    # --- Evaluate ----------------------------------------------------------
    results: dict[str, float] = {}
    for k in sorted(ks):
        results[f"MRR@{k}"] = mrr_at_k(rankings, k)
        results[f"Recall@{k}"] = recall_at_k(rankings, k)
        results[f"MAP@{k}"] = map_at_k(rankings, k)
        results[f"NDCG@{k}"] = ndcg_at_k(rankings, relevance, k)

    logger.info("Retrieval metrics: %s", results)
    return results

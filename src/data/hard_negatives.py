"""Hard negative mining for contrastive code embedding training.

Provides:
- BM25NegativeMiner: Mines hard lexical negatives using BM25 / TF-IDF scoring.
- DenseNegativeMiner: Mines hard semantic negatives using current model embeddings.
- mine_hard_negatives: High-level helper function.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from src.model.encoder import CodeEncoder

logger = logging.getLogger(__name__)


class BM25NegativeMiner:
    """Mine hard lexical negatives using TF-IDF / BM25 style retrieval.

    For each query, computes lexical relevance over the code corpus,
    ranks snippets, and returns a hard negative that is lexically similar
    to the query or ground-truth code but not the ground-truth code itself.

    Parameters
    ----------
    corpus : Sequence[str]
        The collection of code snippets to mine negatives from.
    top_k : int
        Number of top candidates to examine.
    min_rank : int
        Minimum rank cutoff to avoid selecting near-duplicate positives.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        top_k: int = 50,
        min_rank: int = 1,
    ) -> None:
        self.corpus = list(corpus)
        self.top_k = min(top_k, len(self.corpus))
        self.min_rank = min_rank

        logger.info("Building TF-IDF / BM25 index over %d code snippets...", len(self.corpus))
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            max_features=50_000,
            sublinear_tf=True,
        )
        self.corpus_matrix = self.vectorizer.fit_transform(self.corpus)
        logger.info("BM25 index built successfully.")

    def mine_negative_for_query(
        self,
        query: str,
        positive_idx: int | None = None,
    ) -> str:
        """Find a single hard negative snippet for a given query."""
        if not self.corpus:
            return ""

        query_vec = self.vectorizer.transform([query])
        scores = (self.corpus_matrix * query_vec.T).toarray().flatten()

        # Sort descending
        top_indices = np.argsort(-scores)[: self.top_k]

        candidates = [idx for idx in top_indices if positive_idx is None or idx != positive_idx]

        if not candidates:
            # Fallback to random snippet that isn't positive_idx
            alt_indices = [i for i in range(len(self.corpus)) if i != positive_idx]
            chosen_idx = random.choice(alt_indices) if alt_indices else 0
            return self.corpus[chosen_idx]

        # Pick candidate from rank min_rank to top_k
        chosen_idx = candidates[min(self.min_rank, len(candidates) - 1)]
        return self.corpus[chosen_idx]

    def mine_negatives_batch(
        self,
        queries: Sequence[str],
        positive_indices: Sequence[int] | None = None,
    ) -> list[str]:
        """Mine negatives for a batch of queries."""
        negatives: list[str] = []
        for i, q in enumerate(queries):
            pos_idx = positive_indices[i] if positive_indices is not None else None
            negatives.append(self.mine_negative_for_query(q, pos_idx))
        return negatives


class DenseNegativeMiner:
    """Mine hard negatives using dense embeddings from the current model.

    Useful for cross-epoch negative refresh during Stage 2 / Stage 3 training.
    """

    def __init__(self, model: CodeEncoder, corpus: Sequence[str], batch_size: int = 64) -> None:
        self.model = model
        self.corpus = list(corpus)
        self.batch_size = batch_size
        logger.info("Computing dense corpus embeddings for %d snippets...", len(self.corpus))
        self.corpus_embeddings = self.model.encode(
            self.corpus,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )

    def mine_negatives_for_queries(
        self,
        queries: Sequence[str],
        positive_indices: Sequence[int],
        top_k: int = 20,
    ) -> list[str]:
        """Find top dense negative (highest cosine similarity that is not positive)."""
        query_embeddings = self.model.encode(
            list(queries),
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )

        # Dot product because embeddings are unit-norm
        sim_matrix = np.dot(query_embeddings, self.corpus_embeddings.T)

        negatives: list[str] = []
        for q_idx, pos_idx in enumerate(positive_indices):
            scores = sim_matrix[q_idx]
            ranked = np.argsort(-scores)[: top_k + 1]
            # Exclude positive_idx
            chosen_candidates = [idx for idx in ranked if idx != pos_idx]
            chosen_idx = (
                chosen_candidates[0] if chosen_candidates else (pos_idx + 1) % len(self.corpus)
            )
            negatives.append(self.corpus[chosen_idx])

        return negatives


def mine_hard_negatives(
    queries: Sequence[str],
    corpus: Sequence[str],
    positive_indices: Sequence[int] | None = None,
    top_k: int = 50,
) -> list[str]:
    """Helper function to mine BM25 hard negatives for a set of queries."""
    miner = BM25NegativeMiner(corpus=corpus, top_k=top_k)
    return miner.mine_negatives_batch(queries, positive_indices)

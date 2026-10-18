"""Tests for retrieval and embedding quality metric calculations."""

from __future__ import annotations

import numpy as np

from src.evaluation.embedding_metrics import (
    alignment,
    intrinsic_dimensionality,
    isotropy,
    nearest_neighbor_overlap,
    uniformity,
)
from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestRetrievalMetrics:
    """Test standard IR metrics on synthetic rankings."""

    def test_mrr_at_k_perfect(self) -> None:
        # 3 queries; rank 1 is always the target query_idx
        rankings = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        assert mrr_at_k(rankings, k=1) == 1.0
        assert mrr_at_k(rankings, k=5) == 1.0

    def test_mrr_at_k_partial(self) -> None:
        # query 0 at rank 1 (1.0), query 1 at rank 2 (0.5), query 2 missing from top 2 (0.0)
        rankings = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        val = mrr_at_k(rankings, k=2)
        assert np.isclose(val, (1.0 + 0.5 + 0.0) / 3.0)

    def test_recall_at_k(self) -> None:
        rankings = [[0, 1], [0, 1], [0, 1]]
        assert recall_at_k(rankings, k=1) == 1.0 / 3.0
        assert recall_at_k(rankings, k=2) == 2.0 / 3.0

    def test_map_at_k(self) -> None:
        rankings = [[0, 1], [1, 0]]
        assert map_at_k(rankings, k=2) == 1.0

    def test_ndcg_at_k(self) -> None:
        rankings = [[0, 1], [1, 0]]
        relevance = [[1.0, 0.0], [0.0, 1.0]]
        assert ndcg_at_k(rankings, relevance, k=2) == 1.0

    def test_compute_retrieval_metrics_batch(self) -> None:
        # 2 queries, 2 documents
        q_embeds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d_embeds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        metrics = compute_retrieval_metrics(q_embeds, d_embeds, ks=[1, 5])
        assert metrics["MRR@1"] == 1.0
        assert metrics["Recall@1"] == 1.0


class TestEmbeddingQualityMetrics:
    """Test geometric hypersphere properties: alignment, uniformity, isotropy."""

    def test_alignment_identical_pairs(self) -> None:
        e1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        e2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert alignment(e1, e2) == 0.0

    def test_uniformity_range(self, random_embeddings: np.ndarray) -> None:
        u = uniformity(random_embeddings)
        assert isinstance(u, float)
        assert np.isfinite(u)

    def test_isotropy(self, random_embeddings: np.ndarray) -> None:
        iso = isotropy(random_embeddings)
        assert 0.0 <= iso <= 1.0

    def test_intrinsic_dimensionality(self, random_embeddings: np.ndarray) -> None:
        dim = intrinsic_dimensionality(random_embeddings, threshold=0.9)
        assert 1 <= dim <= random_embeddings.shape[1]

    def test_nearest_neighbor_overlap(self) -> None:
        # 4 vectors: first two close (label 0), second two close (label 1)
        emb = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = [0, 0, 1, 1]
        score = nearest_neighbor_overlap(emb, labels, k=1)
        assert score == 1.0

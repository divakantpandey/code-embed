"""Embedding quality metrics.

Measures geometric properties of the learned embedding space such as
alignment, uniformity, isotropy, intrinsic dimensionality, and label-aware
nearest-neighbour overlap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual embedding quality metrics
# ---------------------------------------------------------------------------


def alignment(
    embeddings1: NDArray[np.float32],
    embeddings2: NDArray[np.float32],
) -> float:
    """Mean L2 distance between positive (matched) pairs.

    Lower values indicate that semantically similar items are mapped
    close together in embedding space.

    Parameters
    ----------
    embeddings1:
        Shape ``(n, dim)`` — first element of each positive pair.
    embeddings2:
        Shape ``(n, dim)`` — second element of each positive pair.

    Returns
    -------
    float
        Mean pairwise L2 distance.
    """
    diff = embeddings1 - embeddings2
    distances = np.linalg.norm(diff, axis=1)
    return float(np.mean(distances))


def uniformity(
    embeddings: NDArray[np.float32],
    t: float = 2.0,
) -> float:
    """Measure how uniformly embeddings are distributed on the unit hypersphere.

    Follows the formulation in Wang & Isola (2020):
    ``log E[exp(-t * ||u - v||^2)]`` where *u*, *v* are L2-normalised
    embeddings.  More negative values indicate better uniformity.

    Parameters
    ----------
    embeddings:
        Shape ``(n, dim)``.
    t:
        Temperature parameter.  Default ``2.0``.

    Returns
    -------
    float
        Uniformity score (more negative → more uniform).
    """
    # Normalise to unit sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed: NDArray[np.float32] = embeddings / norms

    # Pairwise squared L2 distances
    sq_dists = cdist(normed, normed, metric="sqeuclidean")

    # Exclude self-pairs by masking the diagonal with inf
    n = sq_dists.shape[0]
    np.fill_diagonal(sq_dists, np.inf)

    # Compute log-mean-exp
    exp_vals = np.exp(-t * sq_dists)
    # Zero out diagonal contributions (exp(-inf) == 0 already but be safe)
    np.fill_diagonal(exp_vals, 0.0)
    n_pairs = n * (n - 1)
    return float(np.log(np.sum(exp_vals) / n_pairs + 1e-30))


def isotropy(embeddings: NDArray[np.float32]) -> float:
    """Variance-ratio measure of isotropy.

    Computes the ratio of the minimum to the maximum eigenvalue of the
    covariance matrix.  A ratio close to 1 means the embedding space is
    isotropic (equally spread in all directions).

    Parameters
    ----------
    embeddings:
        Shape ``(n, dim)``.

    Returns
    -------
    float
        Isotropy score in ``(0, 1]``.
    """
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # numerical safety
    max_eig = eigenvalues[-1]  # eigvalsh returns sorted ascending
    min_eig = eigenvalues[0]
    return float(min_eig / max_eig) if max_eig > 1e-12 else 0.0


def intrinsic_dimensionality(
    embeddings: NDArray[np.float32],
    threshold: float = 0.95,
) -> int:
    """Estimate intrinsic dimensionality via PCA explained-variance ratio.

    Parameters
    ----------
    embeddings:
        Shape ``(n, dim)``.
    threshold:
        Fraction of total variance to retain.  Default ``0.95``.

    Returns
    -------
    int
        Number of principal components needed to explain *threshold*
        fraction of the variance.
    """
    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    # Find first index where cumulative variance exceeds threshold
    dims = int(np.searchsorted(cumulative, threshold) + 1)
    return min(dims, n_components)


def nearest_neighbor_overlap(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    k: int = 10,
) -> float:
    """Fraction of *k*-nearest neighbours that share the same label.

    Parameters
    ----------
    embeddings:
        Shape ``(n, dim)``.
    labels:
        Shape ``(n,)`` — integer class labels for each embedding.
    k:
        Number of nearest neighbours to consider.

    Returns
    -------
    float
        Average overlap ratio across all points, in ``[0, 1]``.
    """
    n = embeddings.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return 0.0

    # Cosine distance → lower is more similar
    dists = cdist(embeddings, embeddings, metric="cosine")
    np.fill_diagonal(dists, np.inf)

    labels_arr = np.asarray(labels)

    # For each point, find k nearest neighbours
    knn_indices = np.argpartition(dists, k, axis=1)[:, :k]

    overlaps: list[float] = []
    for i in range(n):
        neighbour_labels = labels_arr[knn_indices[i]]
        overlap = float(np.sum(neighbour_labels == labels_arr[i]) / k)
        overlaps.append(overlap)

    return float(np.mean(overlaps))


# ---------------------------------------------------------------------------
# Convenience: compute all embedding quality metrics at once
# ---------------------------------------------------------------------------


def compute_embedding_quality(
    embeddings: NDArray[np.float32],
    pos_pairs: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None,
    neg_pairs: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None,
) -> dict[str, float]:
    """Compute a comprehensive set of embedding quality metrics.

    Parameters
    ----------
    embeddings:
        The full embedding matrix, shape ``(n, dim)``.
    pos_pairs:
        Optional tuple ``(anchors, positives)`` each of shape ``(m, dim)``
        for alignment measurement.
    neg_pairs:
        Optional tuple ``(anchors, negatives)`` each of shape ``(p, dim)``.
        Currently used to report mean negative distance for contrast.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    results: dict[str, float] = {}

    logger.info(
        "Computing embedding quality metrics for %d embeddings of dim %d", *embeddings.shape
    )

    # --- Uniformity and isotropy (always computable) -----------------------
    results["uniformity"] = uniformity(embeddings)
    results["isotropy"] = isotropy(embeddings)
    results["intrinsic_dim"] = float(intrinsic_dimensionality(embeddings))

    # --- Alignment (requires positive pairs) -------------------------------
    if pos_pairs is not None:
        anchors, positives = pos_pairs
        results["alignment"] = alignment(anchors, positives)

    # --- Negative distance (optional) --------------------------------------
    if neg_pairs is not None:
        neg_anchors, negatives = neg_pairs
        results["neg_mean_distance"] = alignment(neg_anchors, negatives)

    # --- Alignment / Uniformity gap ----------------------------------------
    if "alignment" in results:
        # Smaller alignment + more negative uniformity → better
        results["au_gap"] = results["alignment"] + results["uniformity"]

    logger.info("Embedding quality metrics: %s", results)
    return results

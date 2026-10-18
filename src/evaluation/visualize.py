"""Visualization utilities for embedding evaluation.

Provides functions to create publication-quality plots:
- UMAP 2-D projection of embeddings
- Cosine-similarity histograms for positive vs. negative pairs
- Training curves (loss, learning-rate, metrics over steps/epochs)

All plots are rendered via *matplotlib* / *seaborn* and saved to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Use non-interactive backend so we can run headlessly
matplotlib.use("Agg")

# Global seaborn style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# UMAP projection
# ---------------------------------------------------------------------------


def plot_umap(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64] | list[str] | None = None,
    save_path: str | Path = "umap_projection.png",
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP Projection of Embeddings",
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 150,
    max_points: int = 10_000,
) -> Path:
    """Create a 2-D UMAP projection scatter plot.

    Parameters
    ----------
    embeddings:
        Shape ``(n, dim)``.
    labels:
        Per-point labels for colouring (int array or string list).
    save_path:
        File path to save the figure.
    n_neighbors:
        UMAP ``n_neighbors`` parameter.
    min_dist:
        UMAP ``min_dist`` parameter.
    title:
        Plot title.
    figsize:
        Figure size in inches.
    dpi:
        Output resolution.
    max_points:
        Down-sample to this many points if the dataset is large.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    import umap  # lazy import — heavy dependency

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional down-sampling
    n = embeddings.shape[0]
    if n > max_points:
        logger.info("Down-sampling from %d to %d points for UMAP", n, max_points)
        indices = np.random.default_rng(42).choice(n, max_points, replace=False)
        embeddings = embeddings[indices]
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = labels[indices]
            else:
                labels = [labels[i] for i in indices]

    logger.info("Fitting UMAP on %d embeddings …", embeddings.shape[0])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels) if isinstance(labels, np.ndarray) else sorted(set(labels))
        palette = sns.color_palette("husl", n_colors=len(unique_labels))
        label_arr = np.asarray(labels)
        for idx, lbl in enumerate(unique_labels):
            mask = label_arr == lbl
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[palette[idx]],
                label=str(lbl),
                s=8,
                alpha=0.6,
            )
        if len(unique_labels) <= 20:
            ax.legend(markerscale=3, fontsize=8, loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    logger.info("UMAP plot saved to %s", save_path)
    return save_path.resolve()


# ---------------------------------------------------------------------------
# Cosine-similarity histogram
# ---------------------------------------------------------------------------


def plot_cosine_histogram(
    pos_sims: NDArray[np.float32],
    neg_sims: NDArray[np.float32],
    save_path: str | Path = "cosine_histogram.png",
    *,
    title: str = "Cosine Similarity Distribution",
    bins: int = 80,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> Path:
    """Plot overlapping histograms of positive and negative cosine similarities.

    Parameters
    ----------
    pos_sims:
        Cosine similarities for positive pairs.
    neg_sims:
        Cosine similarities for negative pairs.
    save_path:
        Output file path.
    title:
        Plot title.
    bins:
        Number of histogram bins.
    figsize:
        Figure size in inches.
    dpi:
        Output resolution.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        neg_sims,
        bins=bins,
        alpha=0.55,
        label=f"Negative (μ={np.mean(neg_sims):.3f})",
        color=sns.color_palette()[3],
        density=True,
    )
    ax.hist(
        pos_sims,
        bins=bins,
        alpha=0.55,
        label=f"Positive (μ={np.mean(pos_sims):.3f})",
        color=sns.color_palette()[0],
        density=True,
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    logger.info("Cosine histogram saved to %s", save_path)
    return save_path.resolve()


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    metrics_history: dict[str, list[float]],
    save_path: str | Path = "training_curves.png",
    *,
    title: str = "Training Curves",
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 150,
    x_key: str | None = None,
) -> Path:
    """Plot training metrics over steps or epochs.

    Parameters
    ----------
    metrics_history:
        Mapping of metric names to lists of values.  A special key
        ``"step"`` or ``"epoch"`` (or *x_key*) is used for the x-axis; if
        absent, the list index is used.
    save_path:
        Output file path.
    title:
        Plot title.
    figsize:
        Figure size in inches.
    dpi:
        Output resolution.
    x_key:
        Key in *metrics_history* to use as the x-axis.  Auto-detected if
        *None*.

    Returns
    -------
    Path
        Absolute path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect x-axis key
    if x_key is None:
        for candidate in ("step", "epoch", "global_step"):
            if candidate in metrics_history:
                x_key = candidate
                break

    x_values = metrics_history.get(x_key) if x_key else None  # type: ignore[arg-type]
    metric_keys = [k for k in metrics_history if k != x_key]

    if not metric_keys:
        logger.warning("No metric keys to plot")
        return save_path.resolve()

    # Separate loss/lr from other metrics for dual-axis readability
    loss_keys = [k for k in metric_keys if "loss" in k.lower()]
    lr_keys = [k for k in metric_keys if "lr" in k.lower() or "learning_rate" in k.lower()]
    other_keys = [k for k in metric_keys if k not in loss_keys and k not in lr_keys]

    # Build subplots
    n_panels = sum(1 for grp in [loss_keys, lr_keys, other_keys] if grp)
    n_panels = max(n_panels, 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]  # type: ignore[list-item]

    ax_idx = 0
    for group, group_label in [
        (loss_keys, "Loss"),
        (lr_keys, "Learning Rate"),
        (other_keys, "Metrics"),
    ]:
        if not group:
            continue
        ax: Any = axes[ax_idx]
        for key in group:
            y = metrics_history[key]
            x = x_values if x_values is not None else list(range(len(y)))
            ax.plot(x[: len(y)], y, label=key, linewidth=1.5)
        ax.set_ylabel(group_label)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    x_label = x_key.replace("_", " ").title() if x_key else "Step"
    axes[-1].set_xlabel(x_label)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    logger.info("Training curves saved to %s", save_path)
    return save_path.resolve()

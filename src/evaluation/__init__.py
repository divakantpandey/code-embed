"""Evaluation benchmarks, metrics, visualization, and reporting."""

from __future__ import annotations

from src.evaluation.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    CodeSearchNetBenchmark,
    CoSQABenchmark,
)
from src.evaluation.embedding_metrics import (
    alignment,
    compute_embedding_quality,
    intrinsic_dimensionality,
    isotropy,
    nearest_neighbor_overlap,
    uniformity,
)
from src.evaluation.report import (
    EvaluationReport,
    generate_json_report,
    generate_markdown_report,
)
from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.visualize import (
    plot_cosine_histogram,
    plot_training_curves,
    plot_umap,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "CodeSearchNetBenchmark",
    "CoSQABenchmark",
    "EvaluationReport",
    "alignment",
    "compute_embedding_quality",
    "compute_retrieval_metrics",
    "generate_json_report",
    "generate_markdown_report",
    "intrinsic_dimensionality",
    "isotropy",
    "map_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "nearest_neighbor_overlap",
    "plot_cosine_histogram",
    "plot_training_curves",
    "plot_umap",
    "recall_at_k",
    "uniformity",
]

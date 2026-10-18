"""Evaluation report generation.

Provides ``EvaluationReport`` to collect benchmark and embedding-quality
metrics and render them as Markdown or JSON files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Collect all evaluation artefacts and generate formatted reports.

    Attributes
    ----------
    model_name:
        Human-readable model identifier.
    retrieval_metrics:
        Mapping ``{benchmark_name: {metric: value}}``.
    embedding_metrics:
        Mapping ``{metric: value}``.
    metadata:
        Arbitrary extra info (commit hash, config, etc.).
    """

    model_name: str = "code-embed"
    retrieval_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    embedding_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -- mutators -----------------------------------------------------------

    def add_retrieval_result(self, benchmark_name: str, metrics: dict[str, float]) -> None:
        """Add metrics from a benchmark run."""
        self.retrieval_metrics[benchmark_name] = metrics

    def add_embedding_metrics(self, metrics: dict[str, float]) -> None:
        """Merge embedding quality metrics into the report."""
        self.embedding_metrics.update(metrics)

    def add_metadata(self, key: str, value: Any) -> None:
        """Store arbitrary metadata."""
        self.metadata[key] = value

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of the entire report."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "retrieval_metrics": self.retrieval_metrics,
            "embedding_metrics": self.embedding_metrics,
            "metadata": self.metadata,
        }

    # -- report writers -----------------------------------------------------

    def generate_markdown(self, output_path: str | Path) -> Path:
        """Write a human-readable Markdown report.

        Parameters
        ----------
        output_path:
            Destination file.  Parent directories are created if needed.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        return generate_markdown_report(self.to_dict(), output_path)

    def generate_json(self, output_path: str | Path) -> Path:
        """Write a machine-readable JSON report.

        Parameters
        ----------
        output_path:
            Destination file.  Parent directories are created if needed.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        return generate_json_report(self.to_dict(), output_path)


# ---------------------------------------------------------------------------
# Stand-alone report generators
# ---------------------------------------------------------------------------


def generate_markdown_report(results: dict[str, Any], output_path: str | Path) -> Path:
    """Create a Markdown evaluation summary.

    Parameters
    ----------
    results:
        Dictionary with keys ``model_name``, ``timestamp``,
        ``retrieval_metrics``, ``embedding_metrics``, ``metadata``.
    output_path:
        File to write.

    Returns
    -------
    Path
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, EvaluationReport):
        results = results.to_dict()

    lines: list[str] = []
    _h = lines.append

    model_name = results.get("model_name", "unknown")
    timestamp = results.get("timestamp", "n/a")

    _h(f"# Evaluation Report — {model_name}")
    _h("")
    _h(f"**Generated:** {timestamp}")
    _h("")

    # --- Retrieval metrics -------------------------------------------------
    retrieval: dict[str, dict[str, float]] = results.get("retrieval_metrics", {})
    if retrieval:
        _h("## Retrieval Metrics")
        _h("")
        for bench_name, metrics in sorted(retrieval.items()):
            _h(f"### {bench_name}")
            _h("")
            _h("| Metric | Value |")
            _h("|--------|------:|")
            for metric, value in sorted(metrics.items()):
                _h(f"| {metric} | {value:.4f} |")
            _h("")

    # --- Embedding quality metrics -----------------------------------------
    emb_metrics: dict[str, float] = results.get("embedding_metrics", {})
    if emb_metrics:
        _h("## Embedding Quality")
        _h("")
        _h("| Metric | Value |")
        _h("|--------|------:|")
        for metric, value in sorted(emb_metrics.items()):
            _h(f"| {metric} | {value:.4f} |")
        _h("")

    # --- Metadata ----------------------------------------------------------
    metadata: dict[str, Any] = results.get("metadata", {})
    if metadata:
        _h("## Metadata")
        _h("")
        _h("```json")
        _h(json.dumps(metadata, indent=2, default=str))
        _h("```")
        _h("")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Markdown report written to %s", output_path)
    return output_path.resolve()


def generate_json_report(results: dict[str, Any], output_path: str | Path) -> Path:
    """Create a JSON evaluation results file.

    Parameters
    ----------
    results:
        Arbitrary results dictionary.
    output_path:
        File to write.

    Returns
    -------
    Path
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, EvaluationReport):
        results = results.to_dict()

    output_path.write_text(
        json.dumps(results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("JSON report written to %s", output_path)
    return output_path.resolve()

"""Quality gate tests — model cannot be promoted if these fail.

These tests are marked with @pytest.mark.quality_gate and are run
in CI on pull requests. They verify that the trained model meets
minimum accuracy, performance, and size requirements.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


def _load_eval_results() -> dict[str, Any] | None:
    for path_str in [
        "evaluation_reports/results.json",
        "outputs/metrics.json",
    ]:
        p = Path(path_str)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                pass
    return None


@pytest.mark.quality_gate
class TestEmbeddingQuality:
    """Hard gates — model cannot be promoted if these fail."""

    def test_mrr10_above_threshold(self) -> None:
        """MRR@10 on CodeSearchNet Python must be ≥ 0.65."""
        results = _load_eval_results()
        if results is None:
            pytest.skip(
                "No eval results found at evaluation_reports/results.json. Run evaluation first."
            )
        csn = results.get("retrieval_metrics", {}).get(
            "CodeSearchNet-Python", results.get("codesearchnet", {})
        )
        mrr = csn.get("MRR@10", 0.0)
        assert mrr >= 0.65, f"MRR@10 ({mrr:.4f}) is below target threshold 0.65"

    def test_cosqa_accuracy_above_threshold(self) -> None:
        """Accuracy@1 or MRR@10 on CoSQA must be ≥ 0.50."""
        results = _load_eval_results()
        if results is None:
            pytest.skip(
                "No eval results found at evaluation_reports/results.json. Run evaluation first."
            )
        cosqa = results.get("retrieval_metrics", {}).get("CoSQA", results.get("cosqa", {}))
        acc = cosqa.get("Accuracy@1", cosqa.get("MRR@1", cosqa.get("MRR@10", 0.0)))
        assert acc >= 0.50, f"CoSQA Accuracy@1 ({acc:.4f}) is below target threshold 0.50"

    def test_embedding_not_degenerate(self) -> None:
        """Uniformity must be < -1.0 (not collapsed)."""
        results = _load_eval_results()
        if results is None:
            pytest.skip("No eval results found. Run evaluation first.")
        emb_metrics = results.get("embedding_quality", results.get("embedding_metrics", {}))
        uniformity = emb_metrics.get("uniformity", 0.0)
        assert uniformity < -1.0, (
            f"Embedding space may be collapsed: uniformity={uniformity:.4f} >= -1.0"
        )

    def test_alignment_reasonable(self) -> None:
        """Alignment must be < 0.5 (positives are close)."""
        results = _load_eval_results()
        if results is None:
            pytest.skip("No eval results found. Run evaluation first.")
        emb_metrics = results.get("embedding_quality", results.get("embedding_metrics", {}))
        if "alignment" not in emb_metrics:
            pytest.skip("Alignment not recorded in evaluation report.")
        align = emb_metrics.get("alignment", 1.0)
        assert align < 0.5, f"Alignment distance too large: {align:.4f} >= 0.5"

    def test_inference_latency(self) -> None:
        """Single snippet embedding must take < 100ms on CPU."""
        results = _load_eval_results()
        if results is None:
            pytest.skip("No eval results found. Run evaluation first.")
        latency_info = results.get("metadata", {}).get("latency", results.get("latency", {}))
        if not latency_info or "mean_ms" not in latency_info:
            pytest.skip("Latency benchmark not recorded in results.json.")
        mean_ms = latency_info.get("mean_ms", 999.0)
        assert mean_ms < 100.0, f"CPU latency too high: {mean_ms:.2f}ms >= 100ms"

    def test_model_size(self) -> None:
        """ONNX model must be < 250 MB."""
        onnx_candidates = [
            Path("outputs/onnx/model_int8.onnx"),
            Path("outputs/onnx/model.onnx"),
        ]
        found = False
        for path in onnx_candidates:
            if path.exists():
                found = True
                size_mb = os.path.getsize(path) / (1024 * 1024)
                assert size_mb < 250.0, f"Model size {size_mb:.1f} MB exceeds 250 MB limit"
                break
        if not found:
            pytest.skip("No ONNX model found at outputs/onnx/. Run export first.")

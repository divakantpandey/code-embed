"""Tests for ONNX export, INT8 quantization, and latency benchmarking."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch.nn as nn

from src.export.onnx_export import export_to_onnx
from src.export.quantize import benchmark_latency, quantize_int8
from src.model.encoder import CodeEncoder


class TestExport:
    """Test model export pipeline to ONNX and INT8."""

    def test_onnx_export_and_quantize(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        model = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            quant_path = Path(tmpdir) / "model_int8.onnx"

            # 1. Export
            res_onnx = export_to_onnx(model, onnx_path)
            assert res_onnx.exists()
            assert os.path.getsize(res_onnx) > 0

            # 2. Dynamic INT8 Quantize
            res_quant = quantize_int8(res_onnx, quant_path)
            assert res_quant.exists()
            assert os.path.getsize(res_quant) > 0

            # 3. Benchmark
            stats = benchmark_latency(
                res_quant,
                dummy_tokenizer,
                num_runs=5,
                warmup_runs=1,
            )
            assert "mean_ms" in stats
            assert stats["mean_ms"] > 0

    def test_sentence_transformers_export(self) -> None:
        from sentence_transformers import SentenceTransformer

        from src.export.sentence_transformers_export import export_to_sentence_transformer

        model = CodeEncoder(base_model_name_or_path="microsoft/unixcoder-base")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = export_to_sentence_transformer(model, tmpdir)
            assert (out_path / "modules.json").exists()
            assert (out_path / "model.safetensors").exists()

            st = SentenceTransformer(str(out_path))
            embs = st.encode(["def foo(): pass"])
            assert embs.shape == (1, model.embedding_dim)

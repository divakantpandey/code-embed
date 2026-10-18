"""ONNX export for CodeEmbed models.

Exports a trained PyTorch CodeEncoder to ONNX format with dynamic batch
and sequence length axes, validates the graph, and verifies numerical parity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.model.encoder import CodeEncoder

logger = logging.getLogger(__name__)


class _ONNXExportWrapper(nn.Module):
    """Wrapper that strips optional arguments for clean ONNX tracing."""

    def __init__(self, model: CodeEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def export_to_onnx(
    model: CodeEncoder,
    output_path: str | Path,
    max_seq_length: int = 512,
    opset_version: int = 18,
) -> Path:
    """Export CodeEncoder to ONNX format.

    Parameters
    ----------
    model : CodeEncoder
        Trained PyTorch CodeEncoder model.
    output_path : str | Path
        Destination .onnx file path.
    max_seq_length : int
        Dummy sequence length used during export.
    opset_version : int
        ONNX opset version (default 18).

    Returns
    -------
    Path
        Absolute path to the exported ONNX model.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = _ONNXExportWrapper(model).cpu()
    wrapper.eval()

    # Dummy inputs for tracing
    dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size"},
    }

    logger.info("Exporting PyTorch model to ONNX at %s...", out_file)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(out_file),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # 1. Check model structure
    onnx_proto = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_proto)
    logger.info("ONNX model structure validated successfully.")

    # 2. Check numerical parity against PyTorch output
    with torch.no_grad():
        pt_out = wrapper(dummy_input_ids, dummy_attention_mask).numpy()

    sess = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    sess_inputs = [inp.name for inp in sess.get_inputs()]
    feed_dict = {}
    if "input_ids" in sess_inputs:
        feed_dict["input_ids"] = dummy_input_ids.numpy()
    if "attention_mask" in sess_inputs:
        feed_dict["attention_mask"] = dummy_attention_mask.numpy()

    ort_out = sess.run(None, feed_dict)[0]

    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
    logger.info("ONNX and PyTorch outputs match within tolerance.")

    # Save tokenizer in same directory
    try:
        model.tokenizer.save_pretrained(str(out_file.parent))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not copy tokenizer to export dir: %s", exc)

    return out_file

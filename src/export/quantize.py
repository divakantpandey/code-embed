"""INT8 quantization and latency benchmarking for ONNX models."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def quantize_int8(
    input_model_path: str | Path,
    output_model_path: str | Path,
    weight_type: QuantType = QuantType.QInt8,
) -> Path:
    """Apply dynamic INT8 quantization to an exported ONNX model.

    Parameters
    ----------
    input_model_path : str | Path
        Path to full precision FP32 ONNX model.
    output_model_path : str | Path
        Destination path for the quantized INT8 model.
    weight_type : QuantType
        Target weight type (default ``QuantType.QInt8``).

    Returns
    -------
    Path
        Path to the quantized model.
    """
    inp = Path(input_model_path).resolve()
    out = Path(output_model_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Applying dynamic INT8 quantization: %s -> %s", inp, out)
    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        weight_type=weight_type,
    )

    fp32_size_mb = os.path.getsize(inp) / (1024 * 1024)
    int8_size_mb = os.path.getsize(out) / (1024 * 1024)
    reduction = (1.0 - (int8_size_mb / fp32_size_mb)) * 100

    logger.info(
        "Quantization complete: %.1f MB -> %.1f MB (%.1f%% reduction)",
        fp32_size_mb,
        int8_size_mb,
        reduction,
    )
    return out


def benchmark_latency(
    model_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    test_snippets: Sequence[str] | None = None,
    num_runs: int = 50,
    warmup_runs: int = 5,
    max_seq_length: int = 512,
) -> dict[str, float]:
    """Measure inference latency per snippet on CPU.

    Parameters
    ----------
    model_path : str | Path
        Path to the ONNX model file.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used to preprocess test snippets.
    test_snippets : Sequence[str] | None
        Sample snippets to benchmark. If None, uses default code snippets.
    num_runs : int
        Number of timed iterations.
    warmup_runs : int
        Number of untimed warmup iterations.
    max_seq_length : int
        Sequence truncation length.

    Returns
    -------
    dict[str, float]
        Latency statistics: mean_ms, median_ms, p95_ms, p99_ms, snippets_per_sec.
    """
    if test_snippets is None or len(test_snippets) == 0:
        test_snippets = [
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
            "import os\nfrom pathlib import Path\ndef find_files(ext):\n    return list(Path('.').rglob(f'*.{ext}'))",
            "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b",
        ]

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # Standard single-core laptop evaluation
    session = ort.InferenceSession(
        str(model_path), sess_options, providers=["CPUExecutionProvider"]
    )

    sess_inputs = [inp.name for inp in session.get_inputs()]

    # Prepare single-item inputs
    inputs_list = []
    for snippet in test_snippets:
        tok = tokenizer(
            snippet,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="np",
        )
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in sess_inputs:
            feed["input_ids"] = tok["input_ids"].astype(np.int64)
        if "attention_mask" in sess_inputs:
            feed["attention_mask"] = tok["attention_mask"].astype(np.int64)
        inputs_list.append(feed)

    # Warmup
    for i in range(warmup_runs):
        inp = inputs_list[i % len(inputs_list)]
        session.run(None, inp)

    # Timed runs
    latencies_sec: list[float] = []
    for i in range(num_runs):
        inp = inputs_list[i % len(inputs_list)]
        t0 = time.perf_counter()
        session.run(None, inp)
        latencies_sec.append(time.perf_counter() - t0)

    latencies_ms = np.array(latencies_sec) * 1000.0

    stats = {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "snippets_per_sec": float(1000.0 / np.mean(latencies_ms)),
    }
    logger.info("Latency benchmark results on CPU: %s", stats)
    return stats

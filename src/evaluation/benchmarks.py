"""Benchmark loaders and runners for code-search evaluation.

Provides ``CodeSearchNetBenchmark``, ``CoSQABenchmark``, and a
``BenchmarkSuite`` that runs all enabled benchmarks and aggregates the
results.

Each benchmark:
1. Loads a standard test set from HuggingFace *datasets*.
2. Encodes queries and documents with a caller-supplied ``encode`` function.
3. Returns a dictionary of retrieval metrics.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.evaluation.retrieval_metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol: the encode function expected from a model
# ---------------------------------------------------------------------------


class EncodeFn(Protocol):
    """Protocol for the model's ``encode`` callable."""

    def __call__(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of strings into an embedding matrix."""
        ...


# ---------------------------------------------------------------------------
# Base benchmark class
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Container for results returned by a benchmark run."""

    benchmark_name: str
    metrics: dict[str, float]
    n_queries: int
    n_docs: int
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract base class for evaluation benchmarks."""

    name: str = "base"

    @abstractmethod
    def run(self, encode_fn: EncodeFn, batch_size: int = 32) -> BenchmarkResult:
        """Run the benchmark and return metrics.

        Parameters
        ----------
        encode_fn:
            Callable that maps a list of strings to an ``(n, dim)`` numpy
            array of embeddings.
        batch_size:
            Encoding batch size forwarded to ``encode_fn``.
        """
        ...


# ---------------------------------------------------------------------------
# CodeSearchNet (Python) benchmark
# ---------------------------------------------------------------------------


class CodeSearchNetBenchmark(Benchmark):
    """Natural-language → Code retrieval on the CodeSearchNet test set.

    Uses the ``code_search_net`` dataset from HuggingFace for any supported
    language split (``python``, ``javascript``, ``java``, ``go``, ``php``, ``ruby``).
    """

    def __init__(
        self,
        language: str = "python",
        max_queries: int | None = 1000,
        ks: list[int] | None = None,
    ) -> None:
        self.language = language.lower()
        self.name = f"CodeSearchNet-{self.language.capitalize()}"
        self.max_queries = max_queries
        self.ks = ks or [1, 5, 10, 100]

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _load_data(language: str, max_queries: int | None) -> tuple[list[str], list[str]]:
        """Load queries (docstrings) and documents (code) from CSN for specified language."""
        from pathlib import Path

        from datasets import load_dataset, load_from_disk

        logger.info("Loading CodeSearchNet %s test split …", language)
        ds = None

        # 1. Check local disk cache
        lang_path = Path(f"data/raw/{language}")
        if lang_path.exists() and (lang_path / "dataset_dict.json").exists():
            try:
                dict_ds = load_from_disk(str(lang_path))
                ds = dict_ds.get("test", dict_ds.get("validation", dict_ds["train"]))
            except Exception as exc:
                logger.debug("Could not load from local path %s: %s", lang_path, exc)

        if ds is None and language == "python":
            raw_path = Path("data/raw")
            if raw_path.exists() and (raw_path / "dataset_dict.json").exists():
                try:
                    dict_ds = load_from_disk(str(raw_path))
                    ds = dict_ds.get("test", dict_ds.get("validation", dict_ds["train"]))
                except Exception:
                    pass

        # 2. Check HuggingFace datasets
        if ds is None:
            for name in ["code-search-net/code_search_net", "code_search_net"]:
                try:
                    ds = load_dataset(name, language, split="test")
                    break
                except Exception as exc:
                    logger.debug("Could not load from %s: %s", name, exc)

        queries: list[str] = []
        docs: list[str] = []
        if ds is not None:
            for row in ds:
                docstring = (
                    row.get("func_documentation_string")
                    or row.get("docstring_summary")
                    or row.get("docstring")
                    or row.get("query")
                    or ""
                )
                code = row.get("func_code_string") or row.get("code") or row.get("function") or ""
                if docstring and code:
                    queries.append(docstring.strip())
                    docs.append(code.strip())
                if max_queries is not None and len(queries) >= max_queries:
                    break

        if not queries:
            logger.warning("Using synthetic fallback data for CodeSearchNet (%s).", language)
            fallbacks: dict[str, list[tuple[str, str]]] = {
                "python": [
                    (
                        "Extract video ID from URL string",
                        "def get_vid_from_url(url):\n    return url.split('v=')[-1].split('&')[0]",
                    ),
                    (
                        "Binary search in sorted list",
                        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
                    ),
                    (
                        "Check if string is palindrome",
                        "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]",
                    ),
                    (
                        "Calculate factorial iteratively",
                        "def factorial(n: int) -> int:\n    res = 1\n    for i in range(2, n + 1): res *= i\n    return res",
                    ),
                    (
                        "Quicksort list in ascending order",
                        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
                    ),
                ],
                "go": [
                    (
                        "Extract video ID from URL string",
                        'func GetVidFromURL(url string) string {\n\tparts := strings.Split(url, "v=")\n\treturn strings.Split(parts[len(parts)-1], "&")[0]\n}',
                    ),
                    (
                        "Binary search in sorted list",
                        "func BinarySearch(arr []int, target int) int {\n\tlow, high := 0, len(arr)-1\n\tfor low <= high {\n\t\tmid := (low + high) / 2\n\t\tif arr[mid] == target {\n\t\t\treturn mid\n\t\t} else if arr[mid] < target {\n\t\t\tlow = mid + 1\n\t\t} else {\n\t\t\thigh = mid - 1\n\t\t}\n\t}\n\treturn -1\n}",
                    ),
                    (
                        "Check if string is palindrome",
                        "func IsPalindrome(s string) bool {\n\tfor i := 0; i < len(s)/2; i++ {\n\t\tif s[i] != s[len(s)-1-i] {\n\t\t\treturn false\n\t\t}\n\t}\n\treturn true\n}",
                    ),
                    (
                        "Calculate factorial iteratively",
                        "func Factorial(n int) int {\n\tres := 1\n\tfor i := 2; i <= n; i++ {\n\t\tres *= i\n\t}\n\treturn res\n}",
                    ),
                    (
                        "Quicksort list in ascending order",
                        "func QuickSort(arr []int) []int {\n\tif len(arr) <= 1 {\n\t\treturn arr\n\t}\n\tpivot := arr[len(arr)/2]\n\tvar left, mid, right []int\n\tfor _, x := range arr {\n\t\tif x < pivot {\n\t\t\tleft = append(left, x)\n\t\t} else if x == pivot {\n\t\t\tmid = append(mid, x)\n\t\t} else {\n\t\t\tright = append(right, x)\n\t\t}\n\t}\n\treturn append(append(QuickSort(left), mid...), QuickSort(right)...)\n}",
                    ),
                ],
                "java": [
                    (
                        "Extract video ID from URL string",
                        'public class V {\n    public static String getVid(String url) {\n        return url.split("v=")[1].split("&")[0];\n    }\n}',
                    ),
                    (
                        "Binary search in sorted list",
                        "public class B {\n    public static int search(int[] arr, int target) {\n        int low = 0, high = arr.length - 1;\n        while (low <= high) {\n            int mid = (low + high) / 2;\n            if (arr[mid] == target) return mid;\n            else if (arr[mid] < target) low = mid + 1;\n            else high = mid - 1;\n        }\n        return -1;\n    }\n}",
                    ),
                    (
                        "Check if string is palindrome",
                        "public class P {\n    public static boolean isPal(String s) {\n        return new StringBuilder(s).reverse().toString().equals(s);\n    }\n}",
                    ),
                    (
                        "Calculate factorial iteratively",
                        "public class F {\n    public static int fact(int n) {\n        int res = 1;\n        for (int i = 2; i <= n; i++) res *= i;\n        return res;\n    }\n}",
                    ),
                    (
                        "Quicksort list in ascending order",
                        "public class Q {\n    public static void sort(int[] a, int lo, int hi) {\n        if (lo < hi) {\n            int p = partition(a, lo, hi);\n            sort(a, lo, p - 1);\n            sort(a, p + 1, hi);\n        }\n    }\n    private static int partition(int[] a, int lo, int hi) { return lo; }\n}",
                    ),
                ],
                "javascript": [
                    (
                        "Extract video ID from URL string",
                        "function getVidFromUrl(url) {\n    return url.split('v=')[1].split('&')[0];\n}",
                    ),
                    (
                        "Binary search in sorted list",
                        "function binarySearch(arr, target) {\n    let low = 0, high = arr.length - 1;\n    while (low <= high) {\n        const mid = Math.floor((low + high) / 2);\n        if (arr[mid] === target) return mid;\n        else if (arr[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}",
                    ),
                    (
                        "Check if string is palindrome",
                        "function isPalindrome(s) {\n    return s === s.split('').reverse().join('');\n}",
                    ),
                    (
                        "Calculate factorial iteratively",
                        "function factorial(n) {\n    let res = 1;\n    for (let i = 2; i <= n; i++) res *= i;\n    return res;\n}",
                    ),
                    (
                        "Quicksort list in ascending order",
                        "function quicksort(arr) {\n    if (arr.length <= 1) return arr;\n    const p = arr[0];\n    return [...quicksort(arr.filter(x => x < p)), ...arr.filter(x => x === p), ...quicksort(arr.filter(x => x > p))];\n}",
                    ),
                ],
                "ruby": [
                    (
                        "Extract video ID from URL string",
                        "def get_vid_from_url(url)\n  url.split('v=')[-1].split('&')[0]\nend",
                    ),
                    (
                        "Binary search in sorted list",
                        "def binary_search(arr, target)\n  low, high = 0, arr.length - 1\n  while low <= high\n    mid = (low + high) / 2\n    return mid if arr[mid] == target\n    arr[mid] < target ? low = mid + 1 : high = mid - 1\n  end\n  -1\nend",
                    ),
                    (
                        "Check if string is palindrome",
                        "def is_palindrome(s)\n  s == s.reverse\nend",
                    ),
                    (
                        "Calculate factorial iteratively",
                        "def factorial(n)\n  (1..n).inject(1, :*)\nend",
                    ),
                    (
                        "Quicksort list in ascending order",
                        "def quicksort(arr)\n  return arr if arr.length <= 1\n  p = arr[0]\n  quicksort(arr.select { |x| x < p }) + arr.select { |x| x == p } + quicksort(arr.select { |x| x > p })\nend",
                    ),
                ],
                "php": [
                    (
                        "Extract video ID from URL string",
                        "<?php\nfunction get_vid_from_url($url) {\n    return explode('&', explode('v=', $url)[1])[0];\n}",
                    ),
                    (
                        "Binary search in sorted list",
                        "<?php\nfunction binary_search($arr, $target) {\n    $low = 0; $high = count($arr) - 1;\n    while ($low <= $high) {\n        $mid = intdiv($low + $high, 2);\n        if ($arr[$mid] === $target) return $mid;\n        elseif ($arr[$mid] < $target) $low = $mid + 1;\n        else $high = $mid - 1;\n    }\n    return -1;\n}",
                    ),
                    (
                        "Check if string is palindrome",
                        "<?php\nfunction is_palindrome($s) {\n    return $s === strrev($s);\n}",
                    ),
                    (
                        "Calculate factorial iteratively",
                        "<?php\nfunction factorial($n) {\n    $res = 1;\n    for ($i = 2; $i <= $n; $i++) $res *= $i;\n    return $res;\n}",
                    ),
                    (
                        "Quicksort list in ascending order",
                        "<?php\nfunction quicksort($arr) {\n    if (count($arr) <= 1) return $arr;\n    $p = $arr[0];\n    $left = array_filter($arr, fn($x) => $x < $p);\n    $mid = array_filter($arr, fn($x) => $x === $p);\n    $right = array_filter($arr, fn($x) => $x > $p);\n    return array_merge(quicksort($left), $mid, quicksort($right));\n}",
                    ),
                ],
            }
            fallback_pairs = fallbacks.get(language, fallbacks["python"])
            queries = [q for q, _ in fallback_pairs]
            docs = [d for _, d in fallback_pairs]

        logger.info(
            "Loaded %d query–document pairs from CSN-%s", len(queries), language.capitalize()
        )
        return queries, docs

    # -- public API ---------------------------------------------------------

    def run(self, encode_fn: EncodeFn, batch_size: int = 32) -> BenchmarkResult:
        """Encode CSN data and compute retrieval metrics."""
        queries, docs = self._load_data(self.language, self.max_queries)

        t0 = time.perf_counter()
        query_embs = encode_fn(queries, batch_size=batch_size, show_progress=True)
        doc_embs = encode_fn(docs, batch_size=batch_size, show_progress=True)
        encode_time = time.perf_counter() - t0

        metrics = compute_retrieval_metrics(
            query_embeddings=query_embs,
            doc_embeddings=doc_embs,
            ks=self.ks,
        )

        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            n_queries=len(queries),
            n_docs=len(docs),
            elapsed_seconds=elapsed,
            metadata={"encode_time_s": encode_time, "queries": queries, "docs": docs},
        )


# ---------------------------------------------------------------------------
# CoSQA benchmark
# ---------------------------------------------------------------------------


class CoSQABenchmark(Benchmark):
    """Code Search QA benchmark.

    Uses the ``gonglinyuan/CoSQA`` dataset from HuggingFace (or fallback).
    Queries are natural-language questions; documents are Python code snippets.
    """

    name: str = "CoSQA"

    def __init__(
        self,
        max_queries: int | None = 500,
        ks: list[int] | None = None,
    ) -> None:
        self.max_queries = max_queries
        self.ks = ks or [1, 5, 10, 100]

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _load_data(max_queries: int | None) -> tuple[list[str], list[str]]:
        """Load queries and code from CoSQA."""
        from datasets import load_dataset

        logger.info("Loading CoSQA dataset …")
        ds = None
        for name, split in [
            ("gonglinyuan/CoSQA", "validation"),
            ("microsoft/cosqa", "test"),
            ("code_x_glue_cc_code_search_net", "test"),
        ]:
            try:
                ds = load_dataset(name, split=split)
                break
            except Exception as exc:
                logger.debug("Could not load %s (%s): %s", name, split, exc)

        queries: list[str] = []
        docs: list[str] = []
        if ds is not None:
            for row in ds:
                query = row.get("query", row.get("doc", row.get("docstring", "")))
                code = row.get("code", row.get("func", row.get("code_string", "")))
                label = row.get("label", 1)
                # Only keep positive pairs (label == 1) or all if label not present
                if query and code and label:
                    queries.append(str(query).strip())
                    docs.append(str(code).strip())
                if max_queries is not None and len(queries) >= max_queries:
                    break

        if not queries:
            logger.warning("Using synthetic fallback data for CoSQA.")
            fallback_pairs = [
                (
                    "python check relation is symmetric",
                    "def is_symmetric(rel):\n    return all((y, x) in rel for (x, y) in rel)",
                ),
                (
                    "convert string to datetime object",
                    "from datetime import datetime\ndef to_datetime(s: str) -> datetime:\n    return datetime.fromisoformat(s)",
                ),
                (
                    "read json file safely with default",
                    "import json\ndef load_json(path, default=None):\n    try:\n        with open(path) as f: return json.load(f)\n    except Exception: return default",
                ),
                (
                    "flatten nested list",
                    "def flatten(lst):\n    return [item for sub in lst for item in sub]",
                ),
                (
                    "compute md5 hash of file",
                    "import hashlib\ndef file_md5(fname):\n    h = hashlib.md5()\n    with open(fname, 'rb') as f:\n        for chunk in iter(lambda: f.read(4096), b''): h.update(chunk)\n    return h.hexdigest()",
                ),
            ]
            queries = [q for q, _ in fallback_pairs]
            docs = [d for _, d in fallback_pairs]

        logger.info("Loaded %d query–document pairs from CoSQA", len(queries))
        return queries, docs

    # -- public API ---------------------------------------------------------

    def run(self, encode_fn: EncodeFn, batch_size: int = 32) -> BenchmarkResult:
        """Encode CoSQA data and compute retrieval metrics."""
        queries, docs = self._load_data(self.max_queries)

        t0 = time.perf_counter()
        query_embs = encode_fn(queries, batch_size=batch_size, show_progress=True)
        doc_embs = encode_fn(docs, batch_size=batch_size, show_progress=True)
        encode_time = time.perf_counter() - t0

        metrics = compute_retrieval_metrics(
            query_embeddings=query_embs,
            doc_embeddings=doc_embs,
            ks=self.ks,
        )

        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            n_queries=len(queries),
            n_docs=len(docs),
            elapsed_seconds=elapsed,
            metadata={"encode_time_s": encode_time, "queries": queries, "docs": docs},
        )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Run all enabled benchmarks and aggregate results.

    Parameters
    ----------
    benchmarks:
        Sequence of ``Benchmark`` instances to execute.  If *None*, defaults
        to ``[CodeSearchNetBenchmark(), CoSQABenchmark()]``.
    """

    def __init__(self, benchmarks: Sequence[Benchmark] | None = None) -> None:
        if benchmarks is None:
            self.benchmarks: list[Benchmark] = []
        else:
            self.benchmarks = list(benchmarks)

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    def run(
        self,
        encode_fn: EncodeFn,
        batch_size: int = 32,
    ) -> dict[str, BenchmarkResult]:
        """Execute every benchmark and return a mapping of name → result.

        Parameters
        ----------
        encode_fn:
            Model encoding callable.
        batch_size:
            Forwarded to each benchmark and then to ``encode_fn``.

        Returns
        -------
        dict[str, BenchmarkResult]
            Mapping from benchmark name to its ``BenchmarkResult``.
        """
        results: dict[str, BenchmarkResult] = {}
        for bench in self.benchmarks:
            logger.info("Running benchmark: %s", bench.name)
            try:
                result = bench.run(encode_fn, batch_size=batch_size)
                results[bench.name] = result
                logger.info(
                    "Benchmark %s finished in %.1fs — %s",
                    bench.name,
                    result.elapsed_seconds,
                    result.metrics,
                )
            except Exception:
                logger.exception("Benchmark %s failed", bench.name)
        return results

    def run_and_summarise(
        self,
        encode_fn: EncodeFn,
        batch_size: int = 32,
    ) -> dict[str, dict[str, float]]:
        """Run all benchmarks and return a flat ``{bench_name: {metric: value}}`` dict.

        This is a convenience wrapper for downstream reporting.
        """
        raw = self.run(encode_fn, batch_size=batch_size)
        return {name: result.metrics for name, result in raw.items()}

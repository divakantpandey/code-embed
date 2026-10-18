"""Standalone evaluation script for benchmarks and quality gates."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.evaluation.benchmarks import (
    BenchmarkSuite,
    CodeSearchNetBenchmark,
    CoSQABenchmark,
)
from src.evaluation.embedding_metrics import compute_embedding_quality
from src.evaluation.report import (
    EvaluationReport,
    generate_json_report,
    generate_markdown_report,
)
from src.evaluation.visualize import plot_umap
from src.model.encoder import CodeEncoder

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Evaluation CLI")


@app.command()
def evaluate(
    config: Path = typer.Option(
        "configs/eval_config.yaml", "--config", "-c", help="Path to evaluation config YAML"
    ),
    model_path: Path = typer.Option(
        "outputs/best_model", "--model", "-m", help="Path to trained model"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Compute device ('auto', 'cpu', 'mps', 'cuda')"
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Limit number of queries per benchmark for fast profiling"
    ),
) -> None:
    """Run full evaluation suite: benchmarks, embedding health, plots, and reports."""
    console.print(f"[bold green]📊 Loading eval config from {config}...[/]")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    # 1. Load model
    console.print(f"[bold blue]Loading model from {model_path}...[/]")
    try:
        model = CodeEncoder.from_pretrained(model_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load trained model from %s (%s). Using base model.", model_path, exc
        )
        base_name = cfg.get("model", {}).get("base", "microsoft/unixcoder-base")
        try:
            model = CodeEncoder(base_model_name_or_path=base_name)
        except Exception:  # noqa: BLE001
            from tests.conftest import DummyTokenizer, DummyTransformerBackbone

            model = CodeEncoder(
                transformer_backbone=DummyTransformerBackbone(64),
                tokenizer=cast(Any, DummyTokenizer()),
                embedding_dim=64,
            )

    model.eval()

    report = EvaluationReport(model_name=str(model_path))

    # 2. Benchmarks
    suite = BenchmarkSuite()
    bench_cfg = cfg.get("benchmarks", {})

    num_q = max_samples if max_samples is not None else 100
    if bench_cfg.get("codesearchnet", {}).get("enabled", True):
        suite.add_benchmark(CodeSearchNetBenchmark(max_queries=num_q))

    if bench_cfg.get("cosqa", {}).get("enabled", True):
        suite.add_benchmark(CoSQABenchmark(max_queries=num_q))

    console.print("[bold cyan]Running retrieval benchmarks...[/]")

    def encode_wrapper(
        texts: list[str], *, batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        res = model.encode(
            texts, batch_size=batch_size, show_progress=show_progress, convert_to_numpy=True
        )
        return np.asarray(res)

    results = suite.run(
        encode_fn=encode_wrapper, batch_size=cfg.get("evaluation", {}).get("batch_size", 64)
    )

    for name, bench_res in results.items():
        report.add_retrieval_result(name, bench_res.metrics)

        table = Table(title=f"Benchmark: {name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in bench_res.metrics.items():
            table.add_row(k, f"{v:.4f}")
        console.print(table)

    # 3. Embedding quality metrics
    console.print("[bold cyan]Computing embedding space geometric metrics...[/]")
    eval_snippets: list[str] = []
    if "CodeSearchNet-Python" in results and "docs" in results["CodeSearchNet-Python"].metadata:
        eval_snippets = results["CodeSearchNet-Python"].metadata["docs"]
    elif "CoSQA" in results and "docs" in results["CoSQA"].metadata:
        eval_snippets = results["CoSQA"].metadata["docs"]
    else:
        eval_snippets = [
            "def compute_sum(a, b): return a + b",
            "def find_max(arr): return max(arr)",
            "def is_empty(seq): return len(seq) == 0",
            "def reverse_string(s): return s[::-1]",
            "class Node:\n    def __init__(self, val):\n        self.val = val",
        ] * 20

    embeddings = model.encode(eval_snippets, convert_to_numpy=True)
    if isinstance(embeddings, np.ndarray) and len(embeddings) > 1:
        emb_metrics = compute_embedding_quality(embeddings)
        report.embedding_metrics = emb_metrics

        table = Table(title="Embedding Quality Metrics")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        for k, v in emb_metrics.items():
            table.add_row(k, f"{v:.4f}")
        console.print(table)

    # 4. Visualizations
    vis_cfg = cfg.get("visualization", {})
    report_dir = Path(cfg.get("report", {}).get("output_dir", "./evaluation_reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    if (
        vis_cfg.get("umap", {}).get("enabled", True)
        and isinstance(embeddings, np.ndarray)
        and len(embeddings) >= 15
    ):
        try:
            plot_path = report_dir / "umap_projection.png"
            plot_umap(embeddings, save_path=plot_path)
            console.print(f"[green]Saved UMAP plot to {plot_path}[/]")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate UMAP plot: %s", exc)

    # 5. Measure CPU latency
    console.print("[bold cyan]Measuring single-snippet CPU latency...[/]")
    test_snippet = (
        "def fibonacci(n: int) -> int:\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    )
    _ = model.encode([test_snippet])
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.encode([test_snippet])
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    report.metadata["latency"] = {"mean_ms": mean_ms, "p95_ms": p95_ms}
    console.print(f"[bold cyan]Single-snippet latency: {mean_ms:.2f} ms (p95: {p95_ms:.2f} ms)[/]")

    # 6. Generate reports
    md_path = report_dir / "report.md"
    json_path = report_dir / "results.json"
    generate_markdown_report(report.to_dict(), md_path)
    generate_json_report(report.to_dict(), json_path)
    console.print(f"[bold green]Saved Markdown report to {md_path}[/]")
    console.print(f"[bold green]Saved JSON metrics to {json_path}[/]")

    # 6. Quality gates check
    console.print("\n[bold magenta]════════════ Quality Gates Check ════════════[/]")
    gates_cfg = cfg.get("quality_gates", {})
    all_passed = True

    csn_metrics = report.retrieval_metrics.get(
        "CodeSearchNet-Python", report.retrieval_metrics.get("codesearchnet", {})
    )
    mrr_target = gates_cfg.get("csn_python_mrr10", 0.65)
    actual_mrr = csn_metrics.get("MRR@10", 0.0)
    passed_mrr = actual_mrr >= mrr_target
    all_passed = all_passed and passed_mrr
    console.print(
        f"CSN MRR@10: {actual_mrr:.4f} (target: ≥{mrr_target:.4f}) -> {'[green]PASS[/]' if passed_mrr else '[red]GATE FAILED (untrained or below target)[/]'}"
    )

    cosqa_metrics = report.retrieval_metrics.get("CoSQA", report.retrieval_metrics.get("cosqa", {}))
    cosqa_target = gates_cfg.get("cosqa_accuracy1", 0.50)
    actual_cosqa = cosqa_metrics.get(
        "Accuracy@1", cosqa_metrics.get("MRR@1", cosqa_metrics.get("MRR@10", 0.0))
    )
    passed_cosqa = actual_cosqa >= cosqa_target
    all_passed = all_passed and passed_cosqa
    console.print(
        f"CoSQA Accuracy@1: {actual_cosqa:.4f} (target: ≥{cosqa_target:.4f}) -> {'[green]PASS[/]' if passed_cosqa else '[red]GATE FAILED (untrained or below target)[/]'}"
    )

    uniformity_target = gates_cfg.get("uniformity_max", -1.0)
    actual_uniformity = report.embedding_metrics.get("uniformity", 0.0)
    passed_uni = actual_uniformity <= uniformity_target
    all_passed = all_passed and passed_uni
    console.print(
        f"Uniformity: {actual_uniformity:.4f} (target: ≤{uniformity_target:.4f}) -> {'[green]PASS[/]' if passed_uni else '[red]GATE FAILED (untrained or above target)[/]'}"
    )

    if not all_passed:
        console.print("[bold red]❌ One or more quality gates failed![/]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]✅ Evaluation complete. All reports written to {report_dir}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

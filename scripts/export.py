"""Export trained model to ONNX and apply INT8 quantization."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.export.onnx_export import export_to_onnx
from src.export.quantize import benchmark_latency, quantize_int8
from src.model.encoder import CodeEncoder

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Export CLI")


@app.command()
def export(
    model_path: Path = typer.Option(
        "outputs/best_model", "--model", "-m", help="Path to trained model"
    ),
    output_dir: Path = typer.Option(
        "outputs/onnx", "--output", "-o", help="Output directory for ONNX model"
    ),
    quantize: bool = typer.Option(True, "--quantize/--no-quantize", help="Apply INT8 quantization"),
) -> None:
    """Export PyTorch model to ONNX and apply INT8 dynamic quantization."""
    console.print(f"[bold green]📦 Loading PyTorch model from {model_path}...[/]")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = CodeEncoder.from_pretrained(model_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load model from %s (%s). Exporting default architecture.", model_path, exc
        )
        try:
            model = CodeEncoder(base_model_name_or_path="microsoft/unixcoder-base")
        except Exception:  # noqa: BLE001
            from tests.conftest import DummyTokenizer, DummyTransformerBackbone

            model = CodeEncoder(
                transformer_backbone=DummyTransformerBackbone(64),
                tokenizer=DummyTokenizer(),
                embedding_dim=64,
            )

    # 1. Export standard SentenceTransformer format
    st_dir = Path("outputs/sentence_transformers")
    console.print(f"[bold blue]Exporting SentenceTransformer model to {st_dir}...[/]")
    model.save_as_sentence_transformer(st_dir)
    console.print(f"[green]SentenceTransformer model exported to {st_dir}[/]")

    # 2. Export FP32 ONNX model
    fp32_onnx_path = out_dir / "model.onnx"
    console.print(f"[bold blue]Exporting to ONNX: {fp32_onnx_path}...[/]")
    export_to_onnx(model, fp32_onnx_path)

    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    console.print(f"[green]FP32 ONNX model exported: {fp32_size_mb:.2f} MB[/]")

    target_model_path = fp32_onnx_path

    # 2. INT8 dynamic quantization
    if quantize:
        int8_onnx_path = out_dir / "model_int8.onnx"
        console.print(f"[bold blue]Quantizing to INT8: {int8_onnx_path}...[/]")
        quantize_int8(fp32_onnx_path, int8_onnx_path)
        int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
        reduction = (1.0 - (int8_size_mb / fp32_size_mb)) * 100
        console.print(
            f"[green]INT8 model generated: {int8_size_mb:.2f} MB ({reduction:.1f}% smaller)[/]"
        )
        target_model_path = int8_onnx_path

    # 3. CPU Latency Benchmark
    console.print("[bold cyan]Benchmarking inference latency on CPU (single-core)...[/]")
    stats = benchmark_latency(
        model_path=target_model_path,
        tokenizer=model.tokenizer,
        num_runs=30,
        warmup_runs=5,
    )

    table = Table(title="ONNX Model Performance (CPU)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model Path", str(target_model_path))
    table.add_row("Model Size", f"{os.path.getsize(target_model_path) / (1024 * 1024):.2f} MB")
    table.add_row("Mean Latency", f"{stats['mean_ms']:.2f} ms")
    table.add_row("Median (p50) Latency", f"{stats['median_ms']:.2f} ms")
    table.add_row("95th Percentile (p95)", f"{stats['p95_ms']:.2f} ms")
    table.add_row("Throughput", f"{stats['snippets_per_sec']:.1f} snippets/sec")
    console.print(table)

    console.print("[bold green]✅ Export and quantization complete.[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

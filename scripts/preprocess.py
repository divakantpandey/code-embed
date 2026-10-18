"""Run full data preprocessing pipeline: dedup, filter, parse, and split."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from rich.console import Console
from rich.logging import RichHandler
from transformers import AutoTokenizer

from src.data.preprocessing import (
    apply_filters,
    deduplicate_minhash,
    preprocess_code,
)

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Data Preprocessing CLI")


@app.command()
def preprocess(
    config: Path = typer.Option(
        "configs/train_config.yaml", "--config", "-c", help="Path to training config YAML"
    ),
    output_dir: Path = typer.Option(
        "data/processed", "--output", "-o", help="Output directory for processed dataset"
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-m", help="Limit number of samples for fast iteration"
    ),
) -> None:
    """Run preprocessing: dedup, quality filter, parse, and export."""
    console.print(f"[bold green]🔧 Loading config from {config}...[/]")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    prep_cfg = cfg.get("data", {}).get("preprocessing", {})
    min_tokens = prep_cfg.get("min_tokens", 10)
    max_tokens = prep_cfg.get("max_tokens", 512)
    dedup_threshold = prep_cfg.get("dedup_jaccard_threshold", 0.8)
    require_function = prep_cfg.get("require_function_def", True)
    require_valid_syntax = prep_cfg.get("require_valid_syntax", True)
    max_comment_ratio = prep_cfg.get("max_comment_ratio", 0.5)

    base_model_name = cfg.get("model", {}).get("base", "microsoft/unixcoder-base")
    console.print(f"[bold blue]Loading tokenizer for {base_model_name}...[/]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load %s tokenizer: %s. Using gpt2 fallback.", base_model_name, exc
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 1. Load dataset
    console.print("[bold blue]Loading CodeSearchNet Python dataset...[/]")
    raw_path = Path("data/raw")
    if raw_path.exists() and (raw_path / "dataset_dict.json").exists():
        ds = load_from_disk(str(raw_path))
    else:
        try:
            ds = load_dataset("code-search-net/code_search_net", "python")
        except Exception:
            try:
                ds = load_dataset("code_search_net", "python")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HuggingFace load failed (%s). Creating synthetic dataset for offline dev.", exc
                )
                synthetic = {
                    "docstring_summary": [
                        "Compute sum of two numbers",
                        "Binary search in sorted list",
                        "Check if string is palindrome",
                        "Bubble sort list of elements",
                    ],
                    "func_code_string": [
                        "def add(a, b):\n    return a + b",
                        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
                        "def is_palindrome(s):\n    return s == s[::-1]",
                        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]: arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
                    ],
                }
                raw_train = Dataset.from_dict(synthetic)
                ds = DatasetDict({"train": raw_train, "test": raw_train, "validation": raw_train})

    train_ds = ds["train"]
    if max_samples and len(train_ds) > max_samples:
        train_ds = train_ds.select(range(max_samples))

    code_col = "func_code_string" if "func_code_string" in train_ds.column_names else "code"

    console.print(f"[bold cyan]Input samples: {len(train_ds)}[/]")

    # 2. MinHash Deduplication
    console.print(f"[bold cyan]Deduplicating with MinHash (threshold={dedup_threshold})...[/]")
    train_deduped = deduplicate_minhash(train_ds, column=code_col, threshold=dedup_threshold)
    console.print(f"[green]After dedup: {len(train_deduped)} samples.[/]")

    # 3. Quality Filtering
    console.print("[bold cyan]Applying quality and syntax filters...[/]")
    filtered_ds = apply_filters(
        dataset=train_deduped,
        tokenizer=tokenizer,
        config=prep_cfg,
        code_column=code_col,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        require_valid_syntax=require_valid_syntax,
        require_function_def=require_function,
        max_comment_ratio=max_comment_ratio,
    )
    console.print(f"[green]After filters: {len(filtered_ds)} samples.[/]")

    # 4. Code normalisation
    console.print("[bold cyan]Normalizing code formatting...[/]")

    def _clean(ex: dict[str, Any]) -> dict[str, Any]:
        ex[code_col] = preprocess_code(ex[code_col])
        return ex

    filtered_ds = filtered_ds.map(_clean)

    # 5. Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filtered_ds.save_to_disk(str(output_path))
    console.print(f"[bold green]✅ Preprocessed dataset saved to {output_path}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

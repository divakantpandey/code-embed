"""Run full data preprocessing pipeline across multiple programming languages.

Includes: deduplication (MinHash LSH), quality filtering, syntax validation (Tree-sitter),
docstring/code parsing, and dataset export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import AutoTokenizer

from scripts.download_data import SYNTHETIC_DATA
from src.data.preprocessing import (
    apply_filters,
    deduplicate_minhash,
    preprocess_code,
)

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="CodeEmbed Multi-Language Data Preprocessing CLI")


def load_raw_language_dataset(lang: str, raw_base_dir: Path) -> Dataset:
    """Load raw dataset for a language from local disk or HuggingFace."""
    lang_dir = None
    if (raw_base_dir / lang).exists():
        lang_dir = raw_base_dir / lang
    elif lang == "python" and (raw_base_dir / "dataset_dict.json").exists():
        lang_dir = raw_base_dir

    if lang_dir and (lang_dir / "dataset_dict.json").exists():
        dict_ds = load_from_disk(str(lang_dir))
        return dict_ds.get("train", dict_ds.get("test", dict_ds["validation"]))

    # Try Hugging Face
    for repo in ["code-search-net/code_search_net", "code_search_net"]:
        try:
            return load_dataset(repo, lang, split="train")
        except Exception:
            pass

    # Fallback synthetic
    synth = SYNTHETIC_DATA.get(lang, SYNTHETIC_DATA["python"])
    return Dataset.from_dict(
        {
            "docstring_summary": synth["docstring_summary"],
            "func_documentation_string": synth["docstring_summary"],
            "func_code_string": synth["func_code_string"],
            "language": [lang] * len(synth["docstring_summary"]),
        }
    )


@app.command()
def preprocess(
    config: Path = typer.Option(
        Path("configs/train_config.yaml"), "--config", "-c", help="Path to training config YAML"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"), "--output", "-o", help="Output directory for processed dataset"
    ),
    languages: str | None = typer.Option(
        None,
        "--languages",
        "-l",
        help="Comma-separated languages (e.g. 'go,java,ruby,php,javascript')",
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-m", help="Limit total number of samples for fast iteration"
    ),
    reuse_cache: bool = typer.Option(
        True,
        "--reuse-cache/--no-reuse-cache",
        help="Reuse existing processed language partitions if valid",
    ),
) -> None:
    """Run preprocessing: dedup, quality filter, parse, and export across languages."""
    console.print(f"[bold green]🔧 Loading config from {config}...[/]")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Determine languages to process
    if languages:
        target_languages = [item.strip().lower() for item in languages.split(",") if item.strip()]
    elif "languages" in cfg:
        target_languages = [item.strip().lower() for item in cfg["languages"]]
    elif "language" in cfg and cfg["language"] != "multilingual":
        target_languages = [cfg["language"].strip().lower()]
    else:
        # Default fallback to python
        target_languages = ["python"]

    console.print(f"[bold cyan]Target languages: {', '.join(target_languages)}[/]")

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

    processed_splits: list[Dataset] = []
    summary_stats: list[dict[str, Any]] = []
    per_lang_limit = (
        max(1, max_samples // len(target_languages)) if max_samples is not None else None
    )

    raw_base = Path("data/raw")
    for lang in target_languages:
        console.print(f"\n[bold magenta]── Processing language: {lang.upper()} ──[/]")
        lang_out = output_dir / lang
        if (
            reuse_cache
            and lang_out.exists()
            and (lang_out / "dataset_info.json").exists()
            and per_lang_limit is None
        ):
            try:
                cached_ds = Dataset.load_from_disk(str(lang_out))
                if len(cached_ds) > 0 and cached_ds[0].get("language") == lang:
                    console.print(
                        f"[bold green]Reusing existing preprocessed {lang.upper()} partition "
                        f"({len(cached_ds)} samples) from {lang_out}[/]"
                    )
                    processed_splits.append(cached_ds)
                    summary_stats.append(
                        {
                            "language": lang,
                            "raw": len(cached_ds),
                            "deduped": len(cached_ds),
                            "final": len(cached_ds),
                        }
                    )
                    continue
            except Exception as e:
                logger.warning("Could not load cached dataset for %s: %s", lang, e)

        raw_ds = load_raw_language_dataset(lang, raw_base)
        initial_count = len(raw_ds)

        if per_lang_limit is not None and initial_count > per_lang_limit:
            raw_ds = raw_ds.select(range(per_lang_limit))

        code_col = "func_code_string" if "func_code_string" in raw_ds.column_names else "code"
        doc_col = (
            "func_documentation_string"
            if "func_documentation_string" in raw_ds.column_names
            else ("docstring_summary" if "docstring_summary" in raw_ds.column_names else "query")
        )

        # Ensure language column exists
        if "language" not in raw_ds.column_names:
            raw_ds = raw_ds.add_column("language", [lang] * len(raw_ds))

        # Standardize column names if needed
        cols_to_keep = [code_col, doc_col, "language"]
        extra_cols = [c for c in raw_ds.column_names if c not in cols_to_keep]
        clean_ds = raw_ds.remove_columns(extra_cols)
        if code_col != "func_code_string":
            clean_ds = clean_ds.rename_column(code_col, "func_code_string")
            code_col = "func_code_string"
        if doc_col != "func_documentation_string":
            clean_ds = clean_ds.rename_column(doc_col, "func_documentation_string")
            doc_col = "func_documentation_string"

        # 1. MinHash Deduplication
        deduped_ds = deduplicate_minhash(clean_ds, column=code_col, threshold=dedup_threshold)

        # 2. Quality Filtering (with per-language syntax and function check)
        filtered_ds = apply_filters(
            dataset=deduped_ds,
            tokenizer=tokenizer,
            config=prep_cfg,
            language=lang,
            code_column=code_col,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            require_valid_syntax=require_valid_syntax,
            require_function_def=require_function,
            max_comment_ratio=max_comment_ratio,
        )

        # 3. Normalization
        def _clean(ex: dict[str, Any]) -> dict[str, Any]:
            ex["func_code_string"] = preprocess_code(ex["func_code_string"])
            return ex

        normalized_ds = filtered_ds.map(_clean)
        processed_splits.append(normalized_ds)

        # Also save individual language split
        lang_out = output_dir / lang
        lang_out.mkdir(parents=True, exist_ok=True)
        normalized_ds.save_to_disk(str(lang_out))

        summary_stats.append(
            {
                "language": lang,
                "raw": initial_count,
                "deduped": len(deduped_ds),
                "final": len(normalized_ds),
            }
        )

    # Combine all language splits into unified dataset
    if len(processed_splits) == 1:
        combined_ds = processed_splits[0]
    else:
        combined_ds = concatenate_datasets(processed_splits).shuffle(seed=42)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_ds.save_to_disk(str(output_dir))

    # Display summary table
    table = Table(title="Multi-Language Preprocessing Summary")
    table.add_column("Language", style="cyan")
    table.add_column("Raw Count", justify="right")
    table.add_column("After Dedup", justify="right")
    table.add_column("Final Filtered", justify="right", style="green")
    for stat in summary_stats:
        table.add_row(
            stat["language"],
            str(stat["raw"]),
            str(stat["deduped"]),
            str(stat["final"]),
        )
    table.add_row(
        "Total",
        str(sum(s["raw"] for s in summary_stats)),
        str(sum(s["deduped"] for s in summary_stats)),
        str(len(combined_ds)),
        style="bold yellow",
    )
    console.print(table)
    console.print(f"[bold green]✅ Preprocessed dataset saved to {output_dir}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""Download and cache all required datasets to disk."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()


def download_codesearchnet(output_dir: Path) -> None:
    """Download CodeSearchNet Python split and save to disk."""
    console.print("[bold blue]Downloading CodeSearchNet (Python)...[/]")
    try:
        ds = load_dataset("code-search-net/code_search_net", "python")
    except Exception:
        try:
            ds = load_dataset("code_search_net", "python")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download CodeSearchNet online (%s).", exc)
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    logger.info("CodeSearchNet Python saved to %s", output_dir)


def download_cosqa() -> None:
    """Download CoSQA dataset."""
    console.print("[bold blue]Downloading CoSQA...[/]")
    try:
        ds = load_dataset("gonglinyuan/CoSQA")
        logger.info("CoSQA loaded: %s", ds)
    except Exception:
        try:
            ds = load_dataset("code_x_glue_cc_code_search", "cosqa")
            logger.info("CoSQA loaded: %s", ds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download CoSQA online (%s).", exc)


def main() -> None:
    """Download all datasets."""
    console.print("[bold green]📦 Downloading datasets...[/]")
    download_codesearchnet(Path("data/raw"))
    download_cosqa()
    console.print("[bold green]✅ Dataset download step complete.[/]")


if __name__ == "__main__":
    main()

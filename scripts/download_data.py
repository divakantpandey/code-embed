"""Download and cache all required datasets."""

from __future__ import annotations

import logging

from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()


def download_codesearchnet() -> None:
    """Download CodeSearchNet Python split."""
    console.print("[bold blue]Downloading CodeSearchNet (Python)...[/]")
    ds = load_dataset("code_search_net", "python")
    logger.info(f"CodeSearchNet Python loaded: {ds}")


def download_cosqa() -> None:
    """Download CoSQA dataset."""
    console.print("[bold blue]Downloading CoSQA...[/]")
    ds = load_dataset("code_x_glue_cc_code_search", "cosqa")
    logger.info(f"CoSQA loaded: {ds}")


def main() -> None:
    """Download all datasets."""
    console.print("[bold green]📦 Downloading datasets...[/]")
    download_codesearchnet()
    download_cosqa()
    console.print("[bold green]✅ All datasets downloaded.[/]")


if __name__ == "__main__":
    main()

"""Interactive demo for the embedding model."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()


def main() -> None:
    """Run interactive demo."""
    console.print("[bold green]🎮 PyCodeEmbed Interactive Demo[/]")
    console.print("Enter Python code snippets to see their embeddings and find similar code.")
    console.print("Type 'quit' to exit.\n")

    # TODO: Implement interactive demo
    # 1. Load ONNX model
    # 2. Accept code snippets from stdin
    # 3. Embed and show vector stats
    # 4. Find similar code from a pre-indexed corpus
    # 5. Show cosine similarity scores

    console.print("[yellow]Demo not yet implemented. See docs/plan.md for details.[/]")


if __name__ == "__main__":
    main()

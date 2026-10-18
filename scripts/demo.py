"""Interactive demo for code search and semantic similarity using PyCodeEmbed."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax

from src.model.encoder import CodeEncoder

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Interactive Demo CLI")

SAMPLE_CORPUS = [
    {
        "name": "Binary Search",
        "description": "Find index of target in sorted array in O(log n) time",
        "code": """def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Find target in sorted array.\"\"\"
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",
    },
    {
        "name": "Quick Sort",
        "description": "Sort list of elements using recursive divide and conquer",
        "code": """def quicksort(arr: list[int]) -> list[int]:
    \"\"\"Sort an array using quicksort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",
    },
    {
        "name": "Palindrome Check",
        "description": "Check if a string or sequence reads the same backwards",
        "code": """def is_palindrome(text: str) -> bool:
    \"\"\"Return True if string is identical when reversed.\"\"\"
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]""",
    },
    {
        "name": "Fibonacci Generator",
        "description": "Generate n fibonacci numbers iteratively",
        "code": """def fibonacci(n: int) -> list[int]:
    \"\"\"Generate the first n Fibonacci numbers.\"\"\"
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fibs = [0, 1]
    for _ in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs""",
    },
    {
        "name": "Read JSON File",
        "description": "Load and parse a JSON file safely from disk",
        "code": """import json
from pathlib import Path

def load_json_file(file_path: str | Path) -> dict:
    \"\"\"Read and parse a JSON file into a Python dict.\"\"\"
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)""",
    },
]


@app.command()
def demo(
    model_path: Path = typer.Option(
        "outputs/best_model", "--model", "-m", help="Path to trained model"
    ),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Run a one-shot query search instead of interactive REPL"
    ),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results to show"),
) -> None:
    """Run interactive semantic code search demo."""
    console.print(
        Panel.fit(
            "[bold cyan]🐍 PyCodeEmbed: Python Code Semantic Search Demo[/]\n"
            "[dim]Lightweight, local-first code embeddings for laptop inference[/]",
            border_style="cyan",
        )
    )

    # 1. Load model
    console.print(f"[bold blue]Loading model ({model_path})...[/]")
    try:
        model = CodeEncoder.from_pretrained(model_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load trained model: %s. Using default architecture.", exc)
        try:
            model = CodeEncoder(base_model_name_or_path="microsoft/unixcoder-base")
        except Exception:  # noqa: BLE001
            from tests.conftest import DummyTokenizer, DummyTransformerBackbone

            model = CodeEncoder(
                transformer_backbone=DummyTransformerBackbone(64),
                tokenizer=DummyTokenizer(),
                embedding_dim=64,
            )

    model.eval()

    # 2. Index corpus
    console.print(f"[bold blue]Indexing {len(SAMPLE_CORPUS)} sample code snippets...[/]")
    corpus_codes = [item["code"] for item in SAMPLE_CORPUS]
    corpus_embeddings = model.encode(corpus_codes, convert_to_numpy=True)
    console.print("[bold green]✅ Indexing complete.[/]\n")

    def _search(q_text: str) -> None:
        t0 = time.perf_counter()
        q_emb = model.encode(q_text, convert_to_numpy=True)
        # Cosine similarity (embeddings are L2 normalized)
        scores = np.dot(corpus_embeddings, q_emb.T).flatten()
        latency_ms = (time.perf_counter() - t0) * 1000

        ranked_indices = np.argsort(-scores)[:top_k]

        console.print(
            f'\n[bold green]Results for query:[/] [yellow]"{q_text}"[/] [dim]({latency_ms:.1f}ms)[/]\n'
        )

        for rank, idx in enumerate(ranked_indices, start=1):
            item = SAMPLE_CORPUS[idx]
            score = float(scores[idx])

            color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
            header = f"Rank {rank}: [bold]{item['name']}[/] — Score: [{color}]{score:.4f}[/]"
            syntax = Syntax(item["code"], "python", theme="monokai", line_numbers=True)
            console.print(
                Panel(syntax, title=header, subtitle=item["description"], border_style="blue")
            )

    # One-shot mode
    if query:
        _search(query)
        return

    # Interactive REPL
    console.print("[bold]Enter a search query in plain English or paste a code snippet.[/]")
    console.print("[dim]Type 'quit' or 'exit' to exit.\n[/]")

    while True:
        try:
            user_input = console.input("[bold cyan]code-embed > [/]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[bold green]Goodbye![/]")
                break
            _search(user_input)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Goodbye![/]")
            break


def main() -> None:
    app()


if __name__ == "__main__":
    main()

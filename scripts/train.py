"""Main training entry point for the 3-stage pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="PyCodeEmbed Training CLI")


@app.command()
def train(
    config: Path = typer.Option(
        "configs/train_config.yaml", "--config", "-c", help="Path to training config YAML"
    ),
    stage: int | None = typer.Option(
        None, "--stage", "-s", help="Run a specific stage only (1, 2, or 3)"
    ),
) -> None:
    """Run the training pipeline."""
    console.print(f"[bold green]🚀 Loading config from {config}...[/]")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    stages_to_run = [stage] if stage else [1, 2, 3]

    for s in stages_to_run:
        stage_key = f"stage{s}"
        stage_cfg = cfg["training"].get(stage_key)
        if stage_cfg and stage_cfg.get("enabled", True):
            console.print(f"[bold blue]📚 Stage {s}: {stage_cfg['method']}[/]")
            # TODO: Implement each training stage
            # - Stage 1: SimCSE contrastive pre-training
            # - Stage 2: Supervised fine-tuning with MNRL
            # - Stage 3: Matryoshka Representation Learning
            logger.info(f"Stage {s} ({stage_cfg['method']}) — not yet implemented")
        else:
            logger.info(f"Stage {s} — skipped (disabled in config)")

    console.print("[bold green]✅ Training complete.[/]")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()

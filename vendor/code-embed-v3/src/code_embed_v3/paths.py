"""Local config overlay.

``configs/`` holds the recipe used for the pilot. A checkout at ``private/`` (or ``$CE3_PRIVATE``) can hold
your own data mix, hyperparameters or prompts under the same relative paths and wins wherever it has a file.
``private/`` is gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def private_root() -> Path:
    return Path(os.environ.get("CE3_PRIVATE", REPO / "private"))


def resolve(rel: str | Path) -> Path:
    """``private/<rel>`` when it exists, else ``<repo>/<rel>``. Absolute paths pass through."""
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    private = private_root() / rel
    return private if private.exists() else REPO / rel

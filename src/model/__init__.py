"""Model architecture: encoder, pooling, projection, and Matryoshka wrapper."""

from __future__ import annotations

from src.model.encoder import CodeEncoder
from src.model.matryoshka import MatryoshkaEncoder
from src.model.pooling import Pooling, PoolingStrategy
from src.model.projection import ProjectionHead

__all__ = [
    "CodeEncoder",
    "MatryoshkaEncoder",
    "Pooling",
    "PoolingStrategy",
    "ProjectionHead",
]

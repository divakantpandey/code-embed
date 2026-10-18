"""Data loading, preprocessing, collation, and negative mining."""

from __future__ import annotations

from src.data.collator import EmbeddingCollator, PairCollator
from src.data.dataset import CodeSearchNetDataset, ContrastiveDataset, TripletDataset
from src.data.hard_negatives import (
    BM25NegativeMiner,
    DenseNegativeMiner,
    mine_hard_negatives,
)
from src.data.preprocessing import (
    apply_filters,
    compute_comment_ratio,
    deduplicate_minhash,
    filter_by_length,
    has_function_definition,
    has_valid_syntax,
    is_auto_generated,
    preprocess_code,
)

__all__ = [
    "BM25NegativeMiner",
    "CodeSearchNetDataset",
    "ContrastiveDataset",
    "DenseNegativeMiner",
    "EmbeddingCollator",
    "PairCollator",
    "TripletDataset",
    "apply_filters",
    "compute_comment_ratio",
    "deduplicate_minhash",
    "filter_by_length",
    "has_function_definition",
    "has_valid_syntax",
    "is_auto_generated",
    "mine_hard_negatives",
    "preprocess_code",
]

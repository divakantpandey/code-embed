"""Dataset classes for the 3-stage code embedding training pipeline.

Provides PyTorch Datasets for:
- SimCSE unsupervised contrastive learning (Stage 1)
- Query-code pair training (Stage 2)
- Triplet training with hard negatives (Stage 2/3)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset[dict[str, str]]):
    """Dataset for Stage 1 SimCSE-style unsupervised contrastive pre-training.

    Each item yields code snippet text. In SimCSE, the model runs two forward
    passes per item with different dropout masks to generate positive pairs.

    Parameters
    ----------
    codes : Sequence[str]
        List of deduplicated and filtered Python code snippets.
    """

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = list(codes)

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> dict[str, str]:
        code = self.codes[idx]
        return {"text": code, "code": code}


class CodeSearchNetDataset(Dataset[dict[str, str]]):
    """Dataset for Stage 2 supervised query–code pair training.

    Yields pairs of natural language queries (docstrings/summaries) and
    their corresponding Python function implementations.

    Parameters
    ----------
    queries : Sequence[str]
        Natural language queries / docstring summaries.
    codes : Sequence[str]
        Python function code snippets.
    """

    def __init__(self, queries: Sequence[str], codes: Sequence[str]) -> None:
        if len(queries) != len(codes):
            raise ValueError(
                f"queries ({len(queries)}) and codes ({len(codes)}) must have the same length"
            )
        self.queries = list(queries)
        self.codes = list(codes)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "query": self.queries[idx],
            "code": self.codes[idx],
        }

    @classmethod
    def from_hf_dataset(
        cls,
        hf_dataset: Any,
        query_column: str = "docstring_summary",
        code_column: str = "func_code_string",
    ) -> CodeSearchNetDataset:
        """Create from a HuggingFace Dataset object."""
        queries: list[str] = []
        codes: list[str] = []
        for example in hf_dataset:
            q = example.get(query_column) or example.get("docstring") or ""
            c = example.get(code_column) or example.get("code") or ""
            if q.strip() and c.strip():
                queries.append(q.strip())
                codes.append(c.strip())
        return cls(queries=queries, codes=codes)


class TripletDataset(Dataset[dict[str, str]]):
    """Dataset for triplet ranking with explicit hard negatives.

    Each item contains an anchor (query), a positive (matching code),
    and a negative (hard negative code).

    Parameters
    ----------
    queries : Sequence[str]
        Anchor queries.
    positives : Sequence[str]
        Ground-truth positive code snippets.
    negatives : Sequence[str]
        Hard negative code snippets.
    """

    def __init__(
        self,
        queries: Sequence[str],
        positives: Sequence[str],
        negatives: Sequence[str],
    ) -> None:
        if not (len(queries) == len(positives) == len(negatives)):
            raise ValueError("queries, positives, and negatives must all have equal length")
        self.queries = list(queries)
        self.positives = list(positives)
        self.negatives = list(negatives)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "query": self.queries[idx],
            "positive": self.positives[idx],
            "negative": self.negatives[idx],
        }

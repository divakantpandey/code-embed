"""Data collators for tokenising and padding batches during training.

Provides :class:`EmbeddingCollator` for single-sequence batches and
:class:`PairCollator` for query–code pair batches used in contrastive
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class EmbeddingCollator:
    """Tokenise and pad a batch of code strings for the encoder model.

    Each example in the batch is expected to be **a single string** (or a dict
    with a ``"text"`` key).  The collator returns a ``dict`` with the usual
    ``input_ids``, ``attention_mask``, and (optionally) ``token_type_ids``
    tensors, ready to be fed into a HuggingFace encoder model.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer instance.
    max_length:
        Maximum sequence length (tokens).  Sequences longer than this will be
        truncated.
    padding:
        Padding strategy – ``"longest"`` (pad to longest in batch) or
        ``"max_length"`` (pad to *max_length*).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    padding: str = "longest"

    def __call__(self, examples: list[str | dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Tokenise and pad *examples*.

        Parameters
        ----------
        examples:
            A list of code strings **or** a list of dicts each containing a
            ``"text"`` key.

        Returns
        -------
        dict[str, torch.Tensor]
            Tokenised batch with ``input_ids`` and ``attention_mask``.
        """
        texts = self._extract_texts(examples)
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(batch)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_texts(examples: list[str | dict[str, Any]]) -> list[str]:
        """Normalise *examples* to a flat list of strings."""
        texts: list[str] = []
        for ex in examples:
            if isinstance(ex, str):
                texts.append(ex)
            elif isinstance(ex, dict):
                texts.append(ex.get("text", ""))
            else:
                texts.append(str(ex))
        return texts


@dataclass
class PairCollator:
    """Tokenise query–code pairs for contrastive training.

    Each example in the batch must be a ``tuple[str, str]`` or a ``dict`` with
    ``"query"`` and ``"code"`` keys.

    The collator returns a ``dict`` with two groups of tensors:

    * ``query_input_ids``, ``query_attention_mask``
    * ``code_input_ids``, ``code_attention_mask``

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer instance.
    max_length:
        Maximum sequence length for **both** query and code.
    padding:
        Padding strategy.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    padding: str = "longest"

    def __call__(self, examples: list[tuple[str, str] | dict[str, str]]) -> dict[str, torch.Tensor]:
        """Tokenise and pad query–code *examples*.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys prefixed with ``query_`` and ``code_`` for each encoder input.
        """
        queries, codes = self._split_pairs(examples)

        query_batch = self.tokenizer(
            queries,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        code_batch = self.tokenizer(
            codes,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        result: dict[str, torch.Tensor] = {}
        for key, tensor in query_batch.items():
            result[f"query_{key}"] = tensor
        for key, tensor in code_batch.items():
            result[f"code_{key}"] = tensor
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _split_pairs(
        examples: list[tuple[str, str] | dict[str, str]],
    ) -> tuple[list[str], list[str]]:
        """Split *examples* into separate query and code lists."""
        queries: list[str] = []
        codes: list[str] = []
        for ex in examples:
            if isinstance(ex, dict):
                queries.append(ex.get("query", ""))
                codes.append(ex.get("code", ""))
            else:
                # Assume tuple / list of (query, code)
                queries.append(ex[0])
                codes.append(ex[1])
        return queries, codes

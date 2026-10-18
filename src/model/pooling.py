"""Pooling strategies for transformer hidden states.

Supports CLS-token, mean-pool, and max-pool over non-padding tokens.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

PoolingStrategy = Literal["cls", "mean", "max"]


class Pooling(nn.Module):
    """Pool transformer hidden states into a single fixed-size vector.

    Parameters
    ----------
    strategy : PoolingStrategy
        One of ``"cls"`` (first token), ``"mean"`` (average over non-padding
        tokens), or ``"max"`` (element-wise max over non-padding tokens).
    """

    VALID_STRATEGIES: tuple[PoolingStrategy, ...] = ("cls", "mean", "max")

    def __init__(self, strategy: PoolingStrategy = "cls") -> None:
        super().__init__()
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown pooling strategy '{strategy}'. Choose from {self.VALID_STRATEGIES}."
            )
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool hidden states to a single vector per sequence.

        Parameters
        ----------
        last_hidden_state : torch.Tensor
            Shape ``(batch, seq_len, hidden_dim)`` — output of the transformer.
        attention_mask : torch.Tensor
            Shape ``(batch, seq_len)`` — 1 for real tokens, 0 for padding.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, hidden_dim)`` — pooled representation.
        """
        if self.strategy == "cls":
            return self._cls_pool(last_hidden_state)
        if self.strategy == "mean":
            return self._mean_pool(last_hidden_state, attention_mask)
        # strategy == "max"
        return self._max_pool(last_hidden_state, attention_mask)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Return the ``[CLS]`` (first) token representation."""
        return last_hidden_state[:, 0, :]

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool over non-padding tokens.

        Expands the attention mask to ``(batch, seq_len, hidden_dim)`` so it
        can be used element-wise, then divides by the number of real tokens
        per sequence (clamped to avoid division by zero).
        """
        # (batch, seq_len, 1) → broadcast to hidden_dim
        mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / token_counts

    @staticmethod
    def _max_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Element-wise max over non-padding tokens.

        Padding positions are filled with a very large negative value before
        taking the max so they never dominate.
        """
        mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        # Replace padding positions with large negative so they lose the max
        hidden = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
        return hidden.max(dim=1).values

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"strategy={self.strategy!r}"

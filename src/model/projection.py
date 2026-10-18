"""MLP projection head for mapping pooled representations to embedding space.

Architecture: Linear(input_dim → hidden_dim) → GELU → Linear(hidden_dim → output_dim)
Optionally L2-normalises the output so embeddings live on the unit hypersphere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with optional L2 normalisation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming pooled hidden states (e.g. 768 for
        ``unixcoder-base``).
    hidden_dim : int
        Width of the intermediate linear layer.
    output_dim : int
        Final embedding dimensionality (e.g. 384).
    normalize : bool
        If ``True``, L2-normalise the output so that every embedding has
        unit norm.  Recommended for contrastive learning with cosine
        similarity.
    dropout : float
        Dropout probability applied after the activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        normalize: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.normalize = normalize

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for linear layers."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project and optionally normalise.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(batch, input_dim)`` — pooled transformer output.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, output_dim)`` — projected (and possibly
            L2-normalised) embeddings.
        """
        projected = self.mlp(hidden_states)
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"normalize={self.normalize}"

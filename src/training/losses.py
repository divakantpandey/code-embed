"""Loss functions for contrastive code embedding training.

Provides three losses for the 3-stage training pipeline:
- InfoNCELoss: temperature-scaled InfoNCE with in-batch negatives (Stage 1 / SimCSE)
- MultipleNegativesRankingLoss: MNRL with optional hard negatives (Stage 2)
- MatryoshkaLoss: wrapper that computes a base loss at multiple embedding dimensions (Stage 3)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """Temperature-scaled InfoNCE loss for self-supervised contrastive learning.

    Used in Stage 1 (SimCSE): each anchor is paired with a positive produced by
    a second forward pass through the same encoder with different dropout masks.
    All other samples in the batch serve as negatives.

    Args:
        temperature: Scaling factor τ for the cosine-similarity logits.
            Lower values sharpen the softmax distribution. Defaults to ``0.05``.
        scale: Alternative scaling factor for similarity logits (where scale = 1.0 / temperature).
    """

    def __init__(
        self,
        temperature: float | None = None,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if scale is not None and temperature is not None:
            raise ValueError("Specify either 'temperature' or 'scale', not both.")
        if scale is not None:
            if scale <= 0:
                raise ValueError(f"scale must be positive, got {scale}")
            self.scale = float(scale)
            self.temperature = 1.0 / self.scale
        elif temperature is not None:
            if temperature <= 0:
                raise ValueError(f"temperature must be positive, got {temperature}")
            self.temperature = float(temperature)
            self.scale = 1.0 / self.temperature
        else:
            self.temperature = 0.05
            self.scale = 20.0

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the InfoNCE loss.

        Args:
            anchor_embeddings: Tensor of shape ``(B, D)`` — anchor representations.
            positive_embeddings: Tensor of shape ``(B, D)`` — positive representations
                (same sentences, different dropout masks in SimCSE).

        Returns:
            Scalar loss tensor.
        """
        # L2-normalise so dot-product == cosine similarity
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)

        # Cosine similarity matrix: (B, B)
        similarity_matrix = torch.mm(anchor_norm, positive_norm.t()) * self.scale

        # Labels: each anchor i should match positive i  →  diagonal
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Cross-entropy over rows (anchor → positive matching)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature}, scale={self.scale})"


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (MNRL) for supervised contrastive learning.

    For each anchor, its paired positive is the target and all other positives in
    the batch act as in-batch negatives.  Optionally, explicit hard negatives are
    concatenated to the candidate set before computing the softmax.

    This mirrors the ``sentence-transformers`` implementation but operates on raw
    embedding tensors for full control inside a custom training loop.

    Args:
        scale: Scaling factor for similarity logits (defaults to 20.0, equivalent to temperature = 0.05).
        temperature: Alternative scaling factor τ (where scale = 1.0 / temperature).
    """

    def __init__(
        self,
        scale: float | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__()
        if scale is not None and temperature is not None:
            raise ValueError("Specify either 'scale' or 'temperature', not both.")
        if temperature is not None:
            if temperature <= 0:
                raise ValueError(f"temperature must be positive, got {temperature}")
            self.temperature = float(temperature)
            self.scale = 1.0 / self.temperature
        elif scale is not None:
            if scale <= 0:
                raise ValueError(f"scale must be positive, got {scale}")
            self.scale = float(scale)
            self.temperature = 1.0 / self.scale
        else:
            self.scale = 20.0
            self.temperature = 0.05

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MNRL loss.

        Args:
            anchor_embeddings: ``(B, D)`` — anchor representations.
            positive_embeddings: ``(B, D)`` — positive representations.
            negative_embeddings: Optional ``(B, N, D)`` or ``(B * N, D)`` — hard negatives.
                If 3-D, ``N`` negatives per anchor are flattened before scoring.

        Returns:
            Scalar loss tensor.
        """
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)

        # Start with in-batch negatives: (B, B) similarity
        candidates = positive_norm  # (B, D)

        if negative_embeddings is not None:
            # Flatten hard negatives to (B*N, D) if 3-D
            if negative_embeddings.dim() == 3:
                negative_embeddings = negative_embeddings.reshape(-1, negative_embeddings.size(-1))
            neg_norm = F.normalize(negative_embeddings, p=2, dim=-1)
            candidates = torch.cat([candidates, neg_norm], dim=0)  # (B + B*N, D)

        # Similarity matrix: (B, B + B*N)
        similarity_matrix = torch.mm(anchor_norm, candidates.t()) * self.scale

        # Labels: the positive for anchor i is at index i in the candidate set
        labels = torch.arange(anchor_norm.size(0), device=similarity_matrix.device)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale}, temperature={self.temperature})"


class MatryoshkaLoss(nn.Module):
    """Matryoshka Representation Learning (MRL) loss wrapper.

    Computes a weighted sum of a base loss evaluated at multiple embedding
    dimensionalities.  This encourages the model to encode useful information
    into the first ``d`` dimensions for every ``d`` in ``dimensions``.

    Args:
        base_loss: The underlying loss module (e.g. ``InfoNCELoss`` or ``MNRL``).
        dimensions: Sequence of truncation widths to evaluate, sorted ascending.
            Defaults to ``[64, 128, 256, 384]``.
        weights: Per-dimension loss weights.  Must be the same length as
            ``dimensions``.  Defaults to ``[0.1, 0.2, 0.3, 0.4]``.
    """

    DEFAULT_DIMENSIONS: list[int] = [64, 128, 256, 384]
    DEFAULT_WEIGHTS: list[float] = [0.1, 0.2, 0.3, 0.4]

    def __init__(
        self,
        base_loss: nn.Module,
        dimensions: Sequence[int] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        if dimensions is None:
            self.dimensions = list(self.DEFAULT_DIMENSIONS)
            self.weights = list(weights or self.DEFAULT_WEIGHTS)
        else:
            self.dimensions = list(dimensions)
            if weights is None:
                if len(self.dimensions) == len(self.DEFAULT_WEIGHTS):
                    self.weights = list(self.DEFAULT_WEIGHTS)
                else:
                    self.weights = [1.0] * len(self.dimensions)
            else:
                self.weights = list(weights)

        if len(self.dimensions) != len(self.weights):
            raise ValueError(
                f"dimensions ({len(self.dimensions)}) and weights ({len(self.weights)}) "
                "must have the same length"
            )

        # Normalise weights so they sum to 1 for interpretable loss magnitudes
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

        logger.info(
            "MatryoshkaLoss initialised — dims=%s, weights=%s, base=%s",
            self.dimensions,
            [f"{w:.3f}" for w in self.weights],
            self.base_loss.__class__.__name__,
        )

    def forward(
        self,
        anchor_embeddings: torch.Tensor | Sequence[torch.Tensor] | dict[int, torch.Tensor],
        positive_embeddings: torch.Tensor | Sequence[torch.Tensor] | dict[int, torch.Tensor],
        negative_embeddings: (
            torch.Tensor | Sequence[torch.Tensor] | dict[int, torch.Tensor] | None
        ) = None,
    ) -> torch.Tensor:
        """Compute the Matryoshka loss across multiple dimensions.

        Args:
            anchor_embeddings: Full-dimension anchor embeddings of shape ``(B, D_full)``,
                or a sequence/dict of embeddings for each target dimension.
            positive_embeddings: Full-dimension positive embeddings of shape ``(B, D_full)``,
                or a sequence/dict of embeddings for each target dimension.
            negative_embeddings: Optional hard negatives passed through to the base loss.

        Returns:
            Scalar weighted-sum loss.
        """
        # Determine execution device from inputs
        if isinstance(anchor_embeddings, torch.Tensor):
            device = anchor_embeddings.device
        elif isinstance(anchor_embeddings, (list, tuple)):
            if len(anchor_embeddings) == 0:
                raise ValueError("anchor_embeddings sequence cannot be empty")
            device = anchor_embeddings[0].device
        elif isinstance(anchor_embeddings, dict):
            if len(anchor_embeddings) == 0:
                raise ValueError("anchor_embeddings dict cannot be empty")
            device = next(iter(anchor_embeddings.values())).device
        else:
            raise TypeError(
                f"Unsupported anchor_embeddings type: {type(anchor_embeddings)}. "
                "Expected torch.Tensor, list of Tensors, or dict of Tensors."
            )

        total_loss: torch.Tensor | None = None

        for i, (dim, weight) in enumerate(zip(self.dimensions, self.weights)):
            # Truncate or select embeddings for the current dimension
            if isinstance(anchor_embeddings, (list, tuple)):
                anchor_trunc = anchor_embeddings[i][..., :dim]
            elif isinstance(anchor_embeddings, dict):
                anchor_trunc = anchor_embeddings[dim][..., :dim]
            elif isinstance(anchor_embeddings, torch.Tensor):
                anchor_trunc = anchor_embeddings[:, :dim]
            else:
                anchor_trunc = torch.as_tensor(anchor_embeddings[i])[..., :dim]

            if isinstance(positive_embeddings, (list, tuple)):
                positive_trunc = positive_embeddings[i][..., :dim]
            elif isinstance(positive_embeddings, dict):
                positive_trunc = positive_embeddings[dim][..., :dim]
            elif isinstance(positive_embeddings, torch.Tensor):
                positive_trunc = positive_embeddings[:, :dim]
            else:
                positive_trunc = torch.as_tensor(positive_embeddings[i])[..., :dim]

            kwargs: dict[str, torch.Tensor] = {
                "anchor_embeddings": anchor_trunc,
                "positive_embeddings": positive_trunc,
            }

            if negative_embeddings is not None:
                if isinstance(negative_embeddings, (list, tuple)):
                    neg_dim = negative_embeddings[i]
                elif isinstance(negative_embeddings, dict):
                    neg_dim = negative_embeddings[dim]
                else:
                    neg_dim = negative_embeddings

                if neg_dim.dim() == 3:
                    kwargs["negative_embeddings"] = neg_dim[:, :, :dim]
                else:
                    kwargs["negative_embeddings"] = neg_dim[:, :dim]

            dim_loss = self.base_loss(**kwargs)
            weighted_loss = weight * dim_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        if total_loss is None:
            return torch.tensor(0.0, device=device)

        return total_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dimensions={self.dimensions}, "
            f"weights={[f'{w:.3f}' for w in self.weights]}, "
            f"base_loss={self.base_loss!r})"
        )

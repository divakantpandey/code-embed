"""Matryoshka Representation Learning (MRL) wrapper for CodeEncoder.

Allows training and inference across multiple nested dimensionalities
(e.g. 64, 128, 256, 384) while preserving unit-norm properties.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.model.encoder import CodeEncoder

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MatryoshkaEncoder(nn.Module):
    """Wrapper that enables Matryoshka representations on top of a CodeEncoder.

    During forward pass in training, returns a dictionary or list of embeddings
    truncated to each target dimension and re-normalized to unit sphere.
    During inference, supports encoding directly to any target dimension.

    Parameters
    ----------
    encoder : CodeEncoder
        The underlying CodeEncoder instance.
    dimensions : Sequence[int]
        Target nested dimensions in ascending order, e.g. ``(64, 128, 256, 384)``.
        All dimensions must be ≤ ``encoder.embedding_dim``.
    """

    def __init__(
        self,
        encoder: CodeEncoder,
        dimensions: Sequence[int] = (64, 128, 256, 384),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        sorted_dims = tuple(sorted(dimensions))

        max_dim = encoder.embedding_dim
        for d in sorted_dims:
            if d > max_dim:
                raise ValueError(
                    f"Matryoshka dimension {d} exceeds encoder embedding_dim {max_dim}"
                )
            if d <= 0:
                raise ValueError(f"Dimension must be positive, got {d}")

        self.dimensions: tuple[int, ...] = sorted_dims

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> list[torch.Tensor] | dict[int, torch.Tensor]:
        """Compute nested embeddings at each Matryoshka dimension.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs tensor of shape ``(batch_size, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask tensor of shape ``(batch_size, seq_len)``.
        token_type_ids : torch.Tensor | None
            Optional token type IDs.
        return_dict : bool
            If ``True``, returns a mapping ``{dim: embedding_tensor}``.
            Otherwise returns a list of embedding tensors ordered by dimension.

        Returns
        -------
        list[torch.Tensor] | dict[int, torch.Tensor]
            Nested representations, each L2-normalized to unit sphere.
        """
        full_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        outputs: dict[int, torch.Tensor] = {}
        for dim in self.dimensions:
            sliced = full_embeddings[:, :dim]
            normed = F.normalize(sliced, p=2, dim=-1)
            outputs[dim] = normed

        if return_dict:
            return outputs
        return [outputs[dim] for dim in self.dimensions]

    # ------------------------------------------------------------------
    # High-level encoding with selectable target dimensionality
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: list[str] | str,
        *,
        target_dim: int | None = None,
        batch_size: int = 32,
        show_progress: bool = False,
        device: str | torch.device | None = None,
        convert_to_numpy: bool = True,
    ) -> NDArray[np.float32] | torch.Tensor:
        """Encode texts with optional truncation to a specific Matryoshka dimension.

        Parameters
        ----------
        texts : list[str] | str
            Texts to encode.
        target_dim : int | None
            Target embedding dimension. If ``None``, uses full embedding_dim.
        batch_size : int
            Batch size for inference.
        show_progress : bool
            Whether to show progress bar.
        device : str | torch.device | None
            Device for execution.
        convert_to_numpy : bool
            Whether to return numpy ndarray.

        Returns
        -------
        np.ndarray | torch.Tensor
            Embeddings of shape ``(num_texts, target_dim or embedding_dim)``.
        """
        full_embeds = self.encoder.encode(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress,
            device=device,
            convert_to_numpy=False,
        )

        if not isinstance(full_embeds, torch.Tensor):
            full_embeds = torch.from_numpy(full_embeds)

        if target_dim is not None:
            if target_dim not in self.dimensions and target_dim > self.encoder.embedding_dim:
                raise ValueError(
                    f"target_dim {target_dim} cannot exceed encoder dim {self.encoder.embedding_dim}"
                )
            sliced = full_embeds[:, :target_dim]
            full_embeds = F.normalize(sliced, p=2, dim=-1)

        if convert_to_numpy:
            return full_embeds.cpu().numpy().astype(np.float32)
        return full_embeds

    # ------------------------------------------------------------------
    # Delegated encoder properties and serialization
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """Return the HuggingFace tokenizer of the underlying encoder."""
        return self.encoder.tokenizer

    @property
    def max_seq_length(self) -> int:
        """Return maximum sequence length of the underlying encoder."""
        return self.encoder.max_seq_length

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality of the underlying encoder."""
        return self.encoder.embedding_dim

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save the underlying encoder to the specified directory."""
        self.encoder.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        dimensions: Sequence[int] = (64, 128, 256, 384),
        **kwargs: Any,
    ) -> MatryoshkaEncoder:
        """Load an encoder from disk and wrap it with MatryoshkaEncoder."""
        encoder = CodeEncoder.from_pretrained(load_directory, **kwargs)
        return cls(encoder=encoder, dimensions=dimensions)

"""Convert CodeEmbed CodeEncoder to standard SentenceTransformer format.

Allows any downstream user or vector database (Chroma, Qdrant, LangChain, LlamaIndex)
to load and run the model via:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("your-username/code-embed")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, Normalize, Pooling, Transformer

if TYPE_CHECKING:
    from src.model.encoder import CodeEncoder

logger = logging.getLogger(__name__)


def to_sentence_transformer(model: CodeEncoder) -> SentenceTransformer:
    """Convert a PyTorch CodeEncoder to an exact, numerically identical SentenceTransformer.

    Parameters
    ----------
    model : CodeEncoder
        The CodeEmbed encoder instance.

    Returns
    -------
    SentenceTransformer
        A fully compatible SentenceTransformer model.
    """
    model.eval()

    # 1. Base Transformer module
    transformer = Transformer(
        model.base_model_name_or_path,
        max_seq_length=model.max_seq_length,
    )

    # Sync position embeddings extension if needed
    if hasattr(transformer.auto_model, "config") and hasattr(transformer.auto_model, "embeddings"):
        current_max = getattr(transformer.auto_model.config, "max_position_embeddings", 512)
        if model.max_seq_length > current_max:
            model._extend_position_embeddings(transformer.auto_model, model.max_seq_length)

    # Load backbone weights
    transformer.auto_model.load_state_dict(model.backbone.state_dict())

    # 2. Pooling module
    pooling = Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode=model.pooling_strategy,
    )

    # 3. Dense layer 1: Linear(768 -> hidden_dim) + GELU
    h_dim = model.projection_hidden_dim
    dense1 = Dense(
        in_features=model.backbone_hidden_size,
        out_features=h_dim,
        bias=True,
        activation_function=torch.nn.modules.activation.GELU(),
    )
    layer0 = model.projection_head.mlp[0]
    assert isinstance(layer0, torch.nn.Linear)
    dense1.linear.weight.data.copy_(layer0.weight.data)
    if layer0.bias is not None and dense1.linear.bias is not None:
        dense1.linear.bias.data.copy_(layer0.bias.data)

    # 4. Dense layer 2: Linear(hidden_dim -> embedding_dim) + Identity
    dense2 = Dense(
        in_features=h_dim,
        out_features=model.embedding_dim,
        bias=True,
        activation_function=torch.nn.Identity(),
    )
    layer3 = model.projection_head.mlp[3]
    assert isinstance(layer3, torch.nn.Linear)
    dense2.linear.weight.data.copy_(layer3.weight.data)
    if layer3.bias is not None and dense2.linear.bias is not None:
        dense2.linear.bias.data.copy_(layer3.bias.data)

    # 5. Normalize module (unit hypersphere)
    norm = Normalize()

    modules = [transformer, pooling, dense1, dense2, norm]
    st = SentenceTransformer(modules=modules)
    st.eval()
    return st


def export_to_sentence_transformer(
    model: CodeEncoder,
    output_dir: str | Path,
) -> Path:
    """Convert CodeEncoder and save in standard SentenceTransformer directory structure.

    Parameters
    ----------
    model : CodeEncoder
        The PyCodeEmbed encoder instance.
    output_dir : str | Path
        Directory where standard SentenceTransformer files will be written.

    Returns
    -------
    Path
        Resolved directory path containing the saved model.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    st = to_sentence_transformer(model)
    st.save(str(out_path))

    logger.info("Successfully exported SentenceTransformer model to %s", out_path)
    return out_path

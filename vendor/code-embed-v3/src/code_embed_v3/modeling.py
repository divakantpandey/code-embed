"""Model construction and loading, including the bidirectional patch for decoder backbones."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer
from transformers import PreTrainedModel
from transformers.masking_utils import create_bidirectional_mask

from code_embed_v3.config import ModelConfig
from code_embed_v3.prompts import model_prompts

# Written next to the saved model; the bidirectional patch is runtime-only, so loaders must reapply it.
META_FILE = "code_embed_v3.json"


def enable_bidirectional_attention(backbone: PreTrainedModel) -> PreTrainedModel:
    """Make a Qwen2/Qwen3-style decoder attend bidirectionally, on this instance only.

    Patching the module-level ``create_causal_mask`` would also flip any teacher or reranker of the
    same architecture loaded in this process, so instead each attention layer's ``is_causal`` flag is
    cleared and the base model receives a precomputed bidirectional mask mapping.
    """
    base = getattr(backbone, "model", backbone)
    config = base.config
    if "sliding_attention" in (getattr(config, "layer_types", None) or []):
        raise NotImplementedError("sliding-window layers are not supported by the bidirectional patch")
    if getattr(base, "_ce3_bidirectional", False):
        return backbone

    n_patched = 0
    for module in base.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
            n_patched += 1
    if n_patched == 0:
        raise ValueError(f"{type(base).__name__} exposes no `is_causal` attention flags to patch")

    causal_forward = base.forward

    def forward(
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, Any] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(attention_mask, dict):
            if inputs_embeds is None:
                inputs_embeds = base.get_input_embeddings()(input_ids)
                input_ids = None
            attention_mask = {
                "full_attention": create_bidirectional_mask(
                    config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask
                )
            }
        kwargs.setdefault("use_cache", False)
        return causal_forward(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    base.forward = forward
    base._ce3_bidirectional = True
    return backbone


def resolve_attn(requested: str | None) -> str | None:
    """``auto`` -> flash-attention 2 via the ``kernels`` package when it can run, else sdpa.

    Same maths as sdpa; the gain is that sentence-transformers then feeds unpadded (flattened)
    sequences, so mixed-length code stops paying for padding tokens.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and importlib.util.find_spec("kernels") is not None:
        return "flash_attention_2"
    return "sdpa"


def build_model(cfg: ModelConfig, device: str | None = None) -> SentenceTransformer:
    """Fresh embedding model from a pretrained backbone (no embedding head weights yet)."""
    model_kwargs: dict[str, Any] = {}
    # The bidirectional patch supplies its own mask mapping, which only sdpa/eager consume as written.
    attn: str | None
    if cfg.bidirectional:
        attn = "sdpa" if cfg.attn_implementation in (None, "auto") else cfg.attn_implementation
    else:
        attn = resolve_attn(cfg.attn_implementation)
    if attn:
        model_kwargs["attn_implementation"] = attn
    model_kwargs["dtype"] = training_dtype(cfg)

    transformer = Transformer(cfg.backbone, max_seq_length=cfg.max_seq_length, model_kwargs=model_kwargs)
    if cfg.bidirectional:
        enable_bidirectional_attention(transformer.auto_model)

    hidden = transformer.get_embedding_dimension()
    if cfg.matryoshka_dims[0] != hidden:
        raise ValueError(
            f"matryoshka_dims[0]={cfg.matryoshka_dims[0]} must equal the backbone hidden size {hidden}; "
            "projecting above the hidden size adds index cost without adding information"
        )
    # Prompt tokens stay in the pool so every runtime (llama.cpp, TEI, ONNX) reproduces the same vector.
    pooling = Pooling(hidden, pooling_mode=cfg.pooling, include_prompt=True)

    model = SentenceTransformer(
        modules=[transformer, pooling, Normalize()],
        prompts=model_prompts(),
        similarity_fn_name="cosine",
        device=device,
        model_card_data=SentenceTransformerModelCardData(
            model_name=cfg.name, license="apache-2.0", language=["en", "code"]
        ),
    )
    return model


def model_meta(cfg: ModelConfig) -> dict[str, Any]:
    return {"bidirectional": cfg.bidirectional, "name": cfg.name}


def save_model(model: SentenceTransformer, path: str | Path, cfg: ModelConfig) -> None:
    path = Path(path)
    model.save(str(path))
    (path / META_FILE).write_text(json.dumps(model_meta(cfg), indent=2))


def load_model(
    path: str | Path, device: str | None = None, attn_implementation: str | None = None, **kwargs: Any
) -> SentenceTransformer:
    """Load a checkpoint produced by this repo, reapplying runtime patches recorded in its metadata.

    ``attn_implementation="auto"`` opts into flash-attention when available (training and mining);
    the default keeps the library's choice, which evaluation uses for bit-stable numbers.
    """
    meta_path = Path(path) / META_FILE
    bidirectional = meta_path.exists() and bool(json.loads(meta_path.read_text()).get("bidirectional"))
    attn = resolve_attn(attn_implementation) if not bidirectional else None
    if attn:
        model_kwargs = {**kwargs.get("model_kwargs", {}), "attn_implementation": attn}
        # flash-attention only takes fp16/bf16, and this repo saves fp32 checkpoints; training callers pass
        # dtype=float32 explicitly and run under autocast.
        if attn == "flash_attention_2":
            model_kwargs.setdefault("dtype", "bfloat16")
        kwargs["model_kwargs"] = model_kwargs
    model = SentenceTransformer(str(path), device=device, **kwargs)
    if bidirectional:
        transformer = model[0]
        if not isinstance(transformer, Transformer):
            raise TypeError(f"{path}: first module is {type(transformer).__name__}, expected Transformer")
        enable_bidirectional_attention(transformer.auto_model)
    return model


def inference_length(model: SentenceTransformer, cfg: ModelConfig) -> int:
    """``cfg.inference_max_seq_length``, else the backbone's position limit."""
    if cfg.inference_max_seq_length is not None:
        return cfg.inference_max_seq_length
    transformer = model[0]
    if not isinstance(transformer, Transformer):
        raise TypeError(f"first module is {type(transformer).__name__}, expected Transformer")
    return int(transformer.auto_model.config.max_position_embeddings)


def training_dtype(cfg: ModelConfig) -> str:
    """Master-weight dtype. Never the checkpoint's: transformers loads that by default, and
    GTE-ModernBERT ships float16, where AdamW's eps underflows and lr-sized updates round away; the
    first dry-run collapsed to random from exactly that. bf16 autocast still runs the compute."""
    return cfg.torch_dtype or "float32"


def init_model(cfg: ModelConfig, init_from: str | None, device: str | None = None) -> SentenceTransformer:
    """Start a stage either from the raw backbone or from a previous stage's checkpoint."""
    if init_from is None:
        return build_model(cfg, device=device)
    return load_model(
        init_from,
        device=device,
        attn_implementation=cfg.attn_implementation,
        model_kwargs={"dtype": training_dtype(cfg)},
    )

"""Bi-encoder wrapper for code embedding models.

Combines a HuggingFace transformer encoder backbone, pooling strategy,
and an MLP projection head with L2 normalisation to map code snippets
into unit-sphere embedding vectors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from src.model.pooling import Pooling, PoolingStrategy
from src.model.projection import ProjectionHead

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CodeEncoder(nn.Module):
    """Bi-encoder model for Python code and query representation.

    Parameters
    ----------
    base_model_name_or_path : str
        HuggingFace model identifier or local path (e.g. ``"microsoft/unixcoder-base"``).
    embedding_dim : int
        Final embedding dimension (e.g. 1536 or 3072).
    projection_hidden_dim : int | None
        Hidden dimension in the projection MLP. If ``None``, defaults to
        ``(encoder_hidden_size + embedding_dim) // 2``.
    pooling : PoolingStrategy
        Pooling method to reduce token embeddings to a sequence embedding
        (``"cls"``, ``"mean"``, or ``"max"``).
    max_seq_length : int
        Maximum sequence length for tokenization.
    freeze_backbone : bool
        If ``True``, freezes the transformer backbone parameters.
    dropout : float
        Dropout probability in the projection head.
    """

    def __init__(
        self,
        base_model_name_or_path: str = "microsoft/unixcoder-base",
        embedding_dim: int = 1536,
        projection_hidden_dim: int | None = None,
        pooling: PoolingStrategy = "cls",
        max_seq_length: int = 8192,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        transformer_backbone: nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__()
        self.base_model_name_or_path = base_model_name_or_path
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout

        # 1. Load transformer backbone or use provided
        self.backbone: Any
        if transformer_backbone is not None:
            self.backbone = transformer_backbone
            backbone_hidden_size = getattr(self.backbone.config, "hidden_size", 768)
        else:
            self.backbone = AutoModel.from_pretrained(base_model_name_or_path)
            backbone_hidden_size = getattr(self.backbone.config, "hidden_size", 768)

        # Extend position embeddings if requested sequence length exceeds backbone default
        if hasattr(self.backbone, "config") and hasattr(self.backbone, "embeddings"):
            current_max = getattr(self.backbone.config, "max_position_embeddings", 512)
            if max_seq_length > current_max:
                self._extend_position_embeddings(self.backbone, max_seq_length)

        self.backbone_hidden_size = backbone_hidden_size

        # 2. Tokenizer (lazily loaded if not provided)
        self._tokenizer: PreTrainedTokenizerBase | None = tokenizer

        # 3. Pooling layer
        self.pooling_layer = Pooling(strategy=pooling)

        # 4. Projection MLP head
        if projection_hidden_dim is None:
            projection_hidden_dim = max((backbone_hidden_size + embedding_dim) // 2, embedding_dim)
        self.projection_hidden_dim = projection_hidden_dim

        self.projection_head = ProjectionHead(
            input_dim=backbone_hidden_size,
            hidden_dim=projection_hidden_dim,
            output_dim=embedding_dim,
            normalize=True,
            dropout=dropout,
        )

        if freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def _extend_position_embeddings(backbone: nn.Module, target_max_length: int) -> None:
        """Extend position embeddings to target_max_length while preserving pretrained weights."""
        config = getattr(backbone, "config", None)
        embeddings = getattr(backbone, "embeddings", None)
        if config is None or embeddings is None:
            return

        current_max = getattr(config, "max_position_embeddings", None)
        pos_emb = getattr(embeddings, "position_embeddings", None)
        if current_max is None or pos_emb is None or current_max >= target_max_length:
            return

        target_size = target_max_length + 2
        orig_weight = pos_emb.weight.data
        current_size = orig_weight.shape[0]

        if current_size < target_size:
            logger.info(
                "Extending position embeddings from %d to %d tokens (preserving pretrained weights)",
                current_size,
                target_size,
            )
            pad_offset = getattr(config, "pad_token_id", 1) + 1
            extra_needed = target_size - current_size
            pos_weights = orig_weight[pad_offset:].unsqueeze(0).permute(0, 2, 1)
            ext_weights = (
                F.interpolate(pos_weights, size=extra_needed, mode="linear", align_corners=True)
                .permute(0, 2, 1)
                .squeeze(0)
            )

            new_full_weights = torch.cat([orig_weight, ext_weights], dim=0)

            embeddings.position_embeddings = nn.Embedding(target_size, config.hidden_size)
            embeddings.position_embeddings.weight.data = new_full_weights
            config.max_position_embeddings = target_size

            if hasattr(embeddings, "token_type_ids"):
                embeddings.register_buffer(
                    "token_type_ids",
                    torch.zeros((1, target_size), dtype=torch.long),
                    persistent=False,
                )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the HuggingFace tokenizer, loading it if not already present."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        if hasattr(self._tokenizer, "model_max_length"):
            self._tokenizer.model_max_length = self.max_seq_length
        return self._tokenizer

    # ------------------------------------------------------------------
    # Backbone control
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the transformer encoder backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Transformer backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the transformer encoder backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Transformer backbone unfrozen.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract L2-normalised sequence embeddings.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs tensor of shape ``(batch_size, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask tensor of shape ``(batch_size, seq_len)``.
        token_type_ids : torch.Tensor | None
            Optional token type IDs.

        Returns
        -------
        torch.Tensor
            Unit-sphere embedding matrix of shape ``(batch_size, embedding_dim)``.
        """
        kwargs: dict[str, Any] = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(input_ids=input_ids, **kwargs)
        last_hidden_state = outputs.last_hidden_state

        pooled = self.pooling_layer(last_hidden_state, attention_mask)
        projected = self.projection_head(pooled)
        return projected

    # ------------------------------------------------------------------
    # High-level encoding API (inference & eval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: list[str] | str,
        *,
        batch_size: int = 32,
        show_progress: bool = False,
        device: str | torch.device | None = None,
        convert_to_numpy: bool = True,
    ) -> NDArray[np.float32] | torch.Tensor:
        """Encode a string or list of strings into embedding vectors.

        Parameters
        ----------
        texts : list[str] | str
            Code snippets or NL queries to encode.
        batch_size : int
            Inference batch size.
        show_progress : bool
            Whether to show progress bar during batching.
        device : str | torch.device | None
            Target device. If ``None``, uses the model's current device.
        convert_to_numpy : bool
            If ``True``, returns a NumPy ndarray. Otherwise returns a PyTorch tensor.

        Returns
        -------
        np.ndarray | torch.Tensor
            Embedding matrix of shape ``(num_texts, embedding_dim)``.
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            empty = np.empty((0, self.embedding_dim), dtype=np.float32)
            return empty if convert_to_numpy else torch.from_numpy(empty)

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device if device is None else torch.device(device)

        all_embeddings: list[torch.Tensor] = []

        iterator = range(0, len(texts), batch_size)
        if (
            show_progress
            and "scalene" not in sys.modules
            and not os.environ.get("SCALENE_PROFILING")
        ):
            try:
                from rich.progress import track

                iterator = track(iterator, description="Encoding texts...")  # type: ignore[assignment]
            except ImportError:
                pass

        for start_idx in iterator:
            batch_texts = texts[start_idx : start_idx + batch_size]
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded_inputs["input_ids"].to(device)
            attention_mask = encoded_inputs["attention_mask"].to(device)
            token_type_ids = encoded_inputs.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            batch_embeds = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            all_embeddings.append(batch_embeds.detach().cpu())

        if was_training:
            self.train()

        embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return embeddings.numpy().astype(np.float32)
        return embeddings

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save the encoder weights, submodules, and configuration."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save backbone and tokenizer
        backbone_dir = save_dir / "backbone"
        self.backbone.save_pretrained(backbone_dir)
        try:
            self.tokenizer.model_max_length = self.max_seq_length
            self.tokenizer.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(backbone_dir)
            for p in (save_dir / "tokenizer_config.json", backbone_dir / "tokenizer_config.json"):
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        tcfg = json.load(f)
                    tcfg["model_max_length"] = self.max_seq_length
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(tcfg, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save tokenizer: %s", exc)

        # 2. Save projection head weights
        torch.save(
            self.projection_head.state_dict(),
            save_dir / "projection_head.pt",
        )

        # 3. Save config metadata
        config_data = {
            "base_model_name_or_path": self.base_model_name_or_path,
            "backbone_hidden_size": self.backbone_hidden_size,
            "embedding_dim": self.embedding_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
            "pooling": self.pooling_strategy,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout_rate,
        }
        with open(save_dir / "code_embed_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        # Also maintain pycode_embed_config.json for backwards compatibility
        with open(save_dir / "pycode_embed_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info("Saved CodeEncoder to %s", save_dir)

    def save_as_sentence_transformer(self, output_dir: str | Path) -> Path:
        """Export this CodeEncoder to standard SentenceTransformer format."""
        from src.export.sentence_transformers_export import export_to_sentence_transformer

        return export_to_sentence_transformer(self, output_dir)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool = False,
        commit_message: str = "Upload CodeEmbed model",
    ) -> str:
        """Push model weights, tokenizer, config, and metadata to Hugging Face Hub."""
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
            )

        hub_url = f"https://huggingface.co/{repo_id}"
        logger.info("Successfully pushed model to %s", hub_url)
        return hub_url

    @classmethod
    def from_pretrained(
        cls, load_directory_or_repo_id: str | Path, *, token: str | None = None
    ) -> CodeEncoder:
        """Load a saved CodeEncoder from local disk or Hugging Face Hub."""
        load_dir = Path(load_directory_or_repo_id)
        if not load_dir.exists():
            from huggingface_hub import snapshot_download

            load_dir = Path(snapshot_download(repo_id=str(load_directory_or_repo_id), token=token))

        config_path = load_dir / "code_embed_config.json"
        if not config_path.exists():
            config_path = load_dir / "pycode_embed_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {load_dir / 'code_embed_config.json'}")

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        backbone_dir = load_dir / "backbone"
        base_path = str(backbone_dir) if backbone_dir.exists() else cfg["base_model_name_or_path"]

        try:
            tok = AutoTokenizer.from_pretrained(str(load_dir))
        except Exception:  # noqa: BLE001
            tok = AutoTokenizer.from_pretrained(cfg["base_model_name_or_path"])

        model = cls(
            base_model_name_or_path=base_path,
            embedding_dim=cfg["embedding_dim"],
            projection_hidden_dim=cfg.get("projection_hidden_dim"),
            pooling=cfg["pooling"],
            max_seq_length=cfg["max_seq_length"],
            dropout=cfg.get("dropout", 0.1),
            tokenizer=tok,
        )

        projection_weights = load_dir / "projection_head.pt"
        if projection_weights.exists():
            state = torch.load(projection_weights, map_location="cpu")
            model.projection_head.load_state_dict(state)

        return model

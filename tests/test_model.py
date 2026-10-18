"""Tests for model architecture: forward pass, output shapes, pooling, and Matryoshka."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.encoder import CodeEncoder
from src.model.matryoshka import MatryoshkaEncoder
from src.model.pooling import Pooling
from src.model.projection import ProjectionHead


class TestPooling:
    """Tests for pooling operations."""

    def test_cls_pooling(self) -> None:
        batch_size, seq_len, hidden_size = 2, 4, 16
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)

        pooler = Pooling(strategy="cls")
        pooled = pooler(hidden, mask)
        assert pooled.shape == (batch_size, hidden_size)
        # Should match index 0
        assert torch.allclose(pooled, hidden[:, 0, :])

    def test_mean_pooling(self) -> None:
        batch_size, seq_len, hidden_size = 2, 3, 4
        hidden = torch.ones(batch_size, seq_len, hidden_size)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

        pooler = Pooling(strategy="mean")
        pooled = pooler(hidden, mask)
        assert pooled.shape == (batch_size, hidden_size)
        assert torch.allclose(pooled, torch.ones(batch_size, hidden_size))

    def test_max_pooling(self) -> None:
        batch_size, _seq_len, hidden_size = 2, 3, 4
        hidden = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [-10.0, -10.0, -10.0, -10.0]],
                [[2.0, 3.0, 4.0, 5.0], [-10.0, -10.0, -10.0, -10.0], [-10.0, -10.0, -10.0, -10.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

        pooler = Pooling(strategy="max")
        pooled = pooler(hidden, mask)
        assert pooled.shape == (batch_size, hidden_size)
        assert torch.allclose(pooled[0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
        assert torch.allclose(pooled[1], torch.tensor([2.0, 3.0, 4.0, 5.0]))

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            Pooling(strategy="invalid")  # type: ignore[arg-type]


class TestProjectionHead:
    """Tests for projection MLP."""

    def test_output_shape_and_norm(self) -> None:
        proj = ProjectionHead(input_dim=64, hidden_dim=32, output_dim=16, normalize=True)
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 16)
        # Unit sphere check
        norms = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestCodeEncoder:
    """Tests for CodeEncoder integration."""

    def test_forward_pass_and_normalization(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        model = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=32,
            pooling="cls",
        )
        input_ids = torch.randint(1, 100, (3, 8))
        attention_mask = torch.ones((3, 8), dtype=torch.long)

        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
        assert embeddings.shape == (3, 32)
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_encode_method(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        model = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=32,
        )
        texts = ["def foo(): pass", "def bar(): return 1"]
        embeds = model.encode(texts, convert_to_numpy=True)
        assert isinstance(embeds, np.ndarray)
        assert embeds.shape == (2, 32)

    def test_freeze_and_unfreeze_backbone(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        model = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=32,
        )
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        for p in model.projection_head.parameters():
            assert p.requires_grad

        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_extend_position_embeddings_8192(self) -> None:
        """Verify position embeddings can be extended to 8192."""
        from transformers import RobertaConfig, RobertaModel

        config = RobertaConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=514,
        )
        backbone = RobertaModel(config)
        assert backbone.config.max_position_embeddings == 514

        CodeEncoder._extend_position_embeddings(backbone, 8192)
        assert backbone.config.max_position_embeddings == 8194
        assert backbone.embeddings.position_embeddings.weight.shape[0] == 8194

        # Test forward pass with 2000 tokens
        input_ids = torch.randint(2, 90, (1, 2000))
        out = backbone(input_ids)
        assert out.last_hidden_state.shape == (1, 2000, 64)


class TestMatryoshkaEncoder:
    """Tests for Matryoshka Representation Learning."""

    def test_matryoshka_dimensions(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        base_encoder = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )
        mrl = MatryoshkaEncoder(base_encoder, dimensions=(16, 32, 64))

        input_ids = torch.randint(1, 100, (2, 8))
        attention_mask = torch.ones((2, 8), dtype=torch.long)

        outputs = mrl(input_ids, attention_mask, return_dict=True)
        assert isinstance(outputs, dict)
        assert list(outputs.keys()) == [16, 32, 64]
        for dim, tensor in outputs.items():
            assert tensor.shape == (2, dim)
            norms = torch.norm(tensor, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_matryoshka_encode_with_target_dim(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        base_encoder = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )
        mrl = MatryoshkaEncoder(base_encoder, dimensions=(16, 32, 64))
        texts = ["def foo(): pass"]
        embed = mrl.encode(texts, target_dim=16, convert_to_numpy=True)
        assert embed.shape == (1, 16)
        assert np.isclose(np.linalg.norm(embed[0]), 1.0, atol=1e-4)

    def test_matryoshka_forward_list_returns_list_of_tensors(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
    ) -> None:
        base_encoder = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )
        mrl = MatryoshkaEncoder(base_encoder, dimensions=(16, 32, 64))

        input_ids = torch.randint(1, 100, (2, 8))
        attention_mask = torch.ones((2, 8), dtype=torch.long)

        outputs = mrl(input_ids, attention_mask, return_dict=False)
        assert isinstance(outputs, list)
        assert len(outputs) == 3
        for expected_dim, tensor in zip((16, 32, 64), outputs):
            assert tensor.shape == (2, expected_dim)

    def test_matryoshka_delegated_properties_and_save(
        self,
        dummy_backbone: nn.Module,
        dummy_tokenizer: Any,
        tmp_path: Any,
    ) -> None:
        base_encoder = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )
        mrl = MatryoshkaEncoder(base_encoder, dimensions=(16, 32, 64))

        assert mrl.embedding_dim == 64
        assert mrl.max_seq_length == base_encoder.max_seq_length
        assert mrl.tokenizer is base_encoder.tokenizer

        save_dir = tmp_path / "mrl_encoder"
        mrl.save_pretrained(save_dir)
        assert (save_dir / "code_embed_config.json").exists()
        assert (save_dir / "pycode_embed_config.json").exists()
        assert (save_dir / "projection_head.pt").exists()

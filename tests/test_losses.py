"""Tests for loss functions: InfoNCELoss, MultipleNegativesRankingLoss, and MatryoshkaLoss."""

from __future__ import annotations

import pytest
import torch

from src.training.losses import InfoNCELoss, MatryoshkaLoss, MultipleNegativesRankingLoss


class TestInfoNCELoss:
    """Tests for InfoNCELoss initialization and forward pass."""

    def test_init_defaults(self) -> None:
        loss_fn = InfoNCELoss()
        assert loss_fn.temperature == pytest.approx(0.05)
        assert loss_fn.scale == pytest.approx(20.0)

    def test_init_with_temperature(self) -> None:
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.temperature == pytest.approx(0.1)
        assert loss_fn.scale == pytest.approx(10.0)

    def test_init_with_scale(self) -> None:
        loss_fn = InfoNCELoss(scale=25.0)
        assert loss_fn.scale == pytest.approx(25.0)
        assert loss_fn.temperature == pytest.approx(0.04)

    def test_init_both_raises(self) -> None:
        with pytest.raises(ValueError, match="Specify either 'temperature' or 'scale'"):
            InfoNCELoss(temperature=0.05, scale=20.0)

    def test_invalid_values_raise(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            InfoNCELoss(temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            InfoNCELoss(temperature=-0.5)
        with pytest.raises(ValueError, match="scale must be positive"):
            InfoNCELoss(scale=0.0)
        with pytest.raises(ValueError, match="scale must be positive"):
            InfoNCELoss(scale=-10.0)

    def test_forward_shape_and_value(self) -> None:
        torch.manual_seed(42)
        loss_fn = InfoNCELoss(temperature=0.05)
        batch_size, dim = 4, 32
        anchors = torch.randn(batch_size, dim)
        positives = anchors + 0.01 * torch.randn(batch_size, dim)

        loss = loss_fn(anchors, positives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_identical_pairs_yield_low_loss(self) -> None:
        torch.manual_seed(42)
        loss_fn = InfoNCELoss(temperature=0.05)
        batch_size, dim = 8, 32
        anchors = torch.randn(batch_size, dim)

        # Identical anchors and positives
        loss_low = loss_fn(anchors, anchors.clone())
        # Random orthogonal-ish positives
        loss_high = loss_fn(anchors, torch.randn(batch_size, dim))
        assert loss_low.item() < loss_high.item()


class TestMultipleNegativesRankingLoss:
    """Tests for MultipleNegativesRankingLoss initialization and forward pass."""

    def test_init_defaults(self) -> None:
        loss_fn = MultipleNegativesRankingLoss()
        assert loss_fn.scale == pytest.approx(20.0)
        assert loss_fn.temperature == pytest.approx(0.05)

    def test_init_with_scale(self) -> None:
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)
        assert loss_fn.scale == pytest.approx(20.0)
        assert loss_fn.temperature == pytest.approx(0.05)

    def test_init_with_temperature(self) -> None:
        loss_fn = MultipleNegativesRankingLoss(temperature=0.02)
        assert loss_fn.temperature == pytest.approx(0.02)
        assert loss_fn.scale == pytest.approx(50.0)

    def test_init_both_raises(self) -> None:
        with pytest.raises(ValueError, match="Specify either 'scale' or 'temperature'"):
            MultipleNegativesRankingLoss(scale=20.0, temperature=0.05)

    def test_invalid_values_raise(self) -> None:
        with pytest.raises(ValueError, match="scale must be positive"):
            MultipleNegativesRankingLoss(scale=0.0)
        with pytest.raises(ValueError, match="scale must be positive"):
            MultipleNegativesRankingLoss(scale=-5.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            MultipleNegativesRankingLoss(temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            MultipleNegativesRankingLoss(temperature=-0.1)

    def test_forward_without_hard_negatives(self) -> None:
        torch.manual_seed(42)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)
        batch_size, dim = 4, 32
        anchors = torch.randn(batch_size, dim)
        positives = anchors + 0.05 * torch.randn(batch_size, dim)

        loss = loss_fn(anchors, positives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_forward_with_2d_hard_negatives(self) -> None:
        torch.manual_seed(42)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)
        batch_size, dim = 4, 32
        anchors = torch.randn(batch_size, dim)
        positives = anchors + 0.05 * torch.randn(batch_size, dim)
        negatives = torch.randn(batch_size * 2, dim)

        loss = loss_fn(anchors, positives, negatives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_forward_with_3d_hard_negatives(self) -> None:
        torch.manual_seed(42)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)
        batch_size, num_neg, dim = 4, 3, 32
        anchors = torch.randn(batch_size, dim)
        positives = anchors + 0.05 * torch.randn(batch_size, dim)
        negatives = torch.randn(batch_size, num_neg, dim)

        loss = loss_fn(anchors, positives, negatives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestMatryoshkaLoss:
    """Tests for MatryoshkaLoss wrapper."""

    def test_init_and_weight_normalization(self) -> None:
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        mrl_loss = MatryoshkaLoss(
            base_loss=base_loss,
            dimensions=[64, 128, 256],
            weights=[1.0, 1.0, 2.0],
        )
        assert mrl_loss.dimensions == [64, 128, 256]
        assert sum(mrl_loss.weights) == pytest.approx(1.0)
        assert mrl_loss.weights == [0.25, 0.25, 0.5]

    def test_dimension_weight_mismatch_raises(self) -> None:
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        with pytest.raises(ValueError, match="must have the same length"):
            MatryoshkaLoss(base_loss=base_loss, dimensions=[64, 128], weights=[1.0])

    def test_forward_pass(self) -> None:
        torch.manual_seed(42)
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        mrl_loss = MatryoshkaLoss(
            base_loss=base_loss,
            dimensions=[16, 32],
            weights=[0.5, 0.5],
        )
        batch_size, dim = 4, 32
        anchors = torch.randn(batch_size, dim)
        positives = anchors + 0.05 * torch.randn(batch_size, dim)

        loss = mrl_loss(anchors, positives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_forward_pass_with_list_input(self) -> None:
        """Test MatryoshkaLoss when anchor/positives are passed as a list of tensors."""
        torch.manual_seed(42)
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        mrl_loss = MatryoshkaLoss(
            base_loss=base_loss,
            dimensions=[16, 32],
            weights=[0.5, 0.5],
        )
        batch_size = 4
        anchors = [torch.randn(batch_size, 16), torch.randn(batch_size, 32)]
        positives = [a + 0.05 * torch.randn_like(a) for a in anchors]

        loss = mrl_loss(anchors, positives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_forward_pass_with_dict_input(self) -> None:
        """Test MatryoshkaLoss when anchor/positives are passed as a dict of tensors."""
        torch.manual_seed(42)
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        mrl_loss = MatryoshkaLoss(
            base_loss=base_loss,
            dimensions=[16, 32],
            weights=[0.5, 0.5],
        )
        batch_size = 4
        anchors = {16: torch.randn(batch_size, 16), 32: torch.randn(batch_size, 32)}
        positives = {k: v + 0.05 * torch.randn_like(v) for k, v in anchors.items()}

        loss = mrl_loss(anchors, positives)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_forward_pass_with_hard_negatives_list_and_tensor(self) -> None:
        """Test MatryoshkaLoss with hard negatives as tensor and list."""
        torch.manual_seed(42)
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        mrl_loss = MatryoshkaLoss(
            base_loss=base_loss,
            dimensions=[16, 32],
            weights=[0.5, 0.5],
        )
        batch_size = 4
        anchors_list = [torch.randn(batch_size, 16), torch.randn(batch_size, 32)]
        positives_list = [a + 0.05 * torch.randn_like(a) for a in anchors_list]
        negatives_list = [torch.randn(batch_size, 16), torch.randn(batch_size, 32)]

        loss = mrl_loss(anchors_list, positives_list, negatives_list)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

        # Also test with full-dim negatives tensor
        negatives_tensor = torch.randn(batch_size, 32)
        loss2 = mrl_loss(anchors_list, positives_list, negatives_tensor)
        assert loss2.ndim == 0
        assert loss2.item() >= 0.0

    def test_matryoshka_encoder_integration_and_backward(self) -> None:
        """Test end-to-end integration: MatryoshkaEncoder forward -> MatryoshkaLoss -> backward."""
        from src.model.encoder import CodeEncoder
        from src.model.matryoshka import MatryoshkaEncoder
        from tests.conftest import DummyTokenizer, DummyTransformerBackbone

        base_encoder = CodeEncoder(
            transformer_backbone=DummyTransformerBackbone(hidden_size=64),
            tokenizer=DummyTokenizer(),
            embedding_dim=64,
        )
        mrl_encoder = MatryoshkaEncoder(base_encoder, dimensions=(16, 32, 64))
        mrl_loss = MatryoshkaLoss(
            base_loss=MultipleNegativesRankingLoss(scale=20.0),
            dimensions=[16, 32, 64],
        )

        input_ids = torch.randint(1, 100, (4, 8))
        attention_mask = torch.ones((4, 8), dtype=torch.long)

        # Standard forward pass of MatryoshkaEncoder returns list of tensors
        q_embeds = mrl_encoder(input_ids, attention_mask)
        c_embeds = mrl_encoder(input_ids, attention_mask)

        assert isinstance(q_embeds, list)
        loss = mrl_loss(q_embeds, c_embeds)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in base_encoder.parameters()
        )
        assert has_grad

"""Tests for CodeEmbeddingTrainer across training stages and model configurations."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.model.encoder import CodeEncoder
from src.model.matryoshka import MatryoshkaEncoder
from src.training.losses import InfoNCELoss, MatryoshkaLoss, MultipleNegativesRankingLoss
from src.training.trainer import CodeEmbeddingTrainer


class SyntheticPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset yielding synthetic query-code pairs for trainer tests."""

    def __init__(self, size: int = 16, seq_len: int = 8) -> None:
        self.size = size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "query_input_ids": torch.randint(1, 100, (self.seq_len,)),
            "query_attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "code_input_ids": torch.randint(1, 100, (self.seq_len,)),
            "code_attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


class SyntheticSimCSEDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset yielding synthetic single-text batches for SimCSE trainer tests."""

    def __init__(self, size: int = 16, seq_len: int = 8) -> None:
        self.size = size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(1, 100, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


def _collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {k: torch.stack([item[k] for item in items], dim=0) for k in keys}


class TestCodeEmbeddingTrainer:
    """Tests for CodeEmbeddingTrainer across Stage 1, 2, and 3."""

    @pytest.fixture
    def base_encoder(self, dummy_backbone: Any, dummy_tokenizer: Any) -> CodeEncoder:
        return CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )

    def test_stage1_simcse_training_step(self, base_encoder: CodeEncoder) -> None:
        dataset = SyntheticSimCSEDataset(size=8, seq_len=8)
        loader = DataLoader(dataset, batch_size=4, collate_fn=_collate_batch)
        loss_fn = InfoNCELoss(temperature=0.05)

        trainer = CodeEmbeddingTrainer(
            model=base_encoder,
            loss_fn=loss_fn,
            train_dataloader=loader,
            epochs=1,
            device="cpu",
            fp16=False,
        )
        metrics = trainer.train()
        assert "loss" in metrics
        assert metrics["loss"] >= 0.0

    def test_stage2_mnrl_training_step(self, base_encoder: CodeEncoder) -> None:
        dataset = SyntheticPairDataset(size=8, seq_len=8)
        loader = DataLoader(dataset, batch_size=4, collate_fn=_collate_batch)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        trainer = CodeEmbeddingTrainer(
            model=base_encoder,
            loss_fn=loss_fn,
            train_dataloader=loader,
            epochs=1,
            device="cpu",
            fp16=False,
        )
        metrics = trainer.train()
        assert "loss" in metrics
        assert metrics["loss"] >= 0.0

    def test_stage3_matryoshka_training_step(self, base_encoder: CodeEncoder) -> None:
        """Stage 3: MatryoshkaEncoder as model, MatryoshkaLoss as loss_fn."""
        dimensions = [16, 32, 64]
        weights = [0.2, 0.3, 0.5]
        mrl_encoder = MatryoshkaEncoder(base_encoder, dimensions=dimensions)
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        loss_fn = MatryoshkaLoss(base_loss=base_loss, dimensions=dimensions, weights=weights)

        dataset = SyntheticPairDataset(size=8, seq_len=8)
        loader = DataLoader(dataset, batch_size=4, collate_fn=_collate_batch)

        trainer = CodeEmbeddingTrainer(
            model=mrl_encoder,
            loss_fn=loss_fn,
            train_dataloader=loader,
            epochs=1,
            device="cpu",
            fp16=False,
        )
        metrics = trainer.train()
        assert "loss" in metrics
        assert metrics["loss"] >= 0.0

    def test_checkpoint_saving_with_matryoshka_encoder(
        self, base_encoder: CodeEncoder, tmp_path: Any
    ) -> None:
        """Verify checkpoint saving with MatryoshkaEncoder saves encoder files."""
        mrl_encoder = MatryoshkaEncoder(base_encoder, dimensions=[16, 32, 64])
        base_loss = MultipleNegativesRankingLoss(scale=20.0)
        loss_fn = MatryoshkaLoss(base_loss=base_loss, dimensions=[16, 32, 64])

        dataset = SyntheticPairDataset(size=4, seq_len=8)
        loader = DataLoader(dataset, batch_size=4, collate_fn=_collate_batch)

        trainer = CodeEmbeddingTrainer(
            model=mrl_encoder,
            loss_fn=loss_fn,
            train_dataloader=loader,
            device="cpu",
            fp16=False,
        )
        ckpt_dir = tmp_path / "stage3_checkpoint"
        trainer.save_checkpoint(ckpt_dir)

        assert (ckpt_dir / "code_embed_config.json").exists()
        assert (ckpt_dir / "pycode_embed_config.json").exists()
        assert (ckpt_dir / "projection_head.pt").exists()
        assert (ckpt_dir / "optimizer.pt").exists()
        assert (ckpt_dir / "scheduler.pt").exists()

    def test_evaluation_with_eval_dataloader(self, base_encoder: CodeEncoder) -> None:
        """Verify that evaluate() computes true non-zero validation loss when eval_dataloader is provided."""
        train_ds = SyntheticPairDataset(size=8, seq_len=8)
        val_ds = SyntheticPairDataset(size=4, seq_len=8)
        train_loader = DataLoader(train_ds, batch_size=4, collate_fn=_collate_batch)
        val_loader = DataLoader(val_ds, batch_size=4, collate_fn=_collate_batch)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        trainer = CodeEmbeddingTrainer(
            model=base_encoder,
            loss_fn=loss_fn,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            epochs=1,
            device="cpu",
            fp16=False,
        )
        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert "loss" in metrics
        assert metrics["eval_loss"] > 0.0
        assert metrics["loss"] == metrics["eval_loss"]

    def test_evaluation_with_eval_dataloader_and_eval_fn(
        self, base_encoder: CodeEncoder
    ) -> None:
        """Verify that evaluate() merges both validation loss and custom eval_fn metrics."""
        train_ds = SyntheticPairDataset(size=8, seq_len=8)
        val_ds = SyntheticPairDataset(size=4, seq_len=8)
        train_loader = DataLoader(train_ds, batch_size=4, collate_fn=_collate_batch)
        val_loader = DataLoader(val_ds, batch_size=4, collate_fn=_collate_batch)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        def mock_eval_fn() -> dict[str, float]:
            return {"eval/MRR@10": 0.75, "eval/NDCG@10": 0.85}

        trainer = CodeEmbeddingTrainer(
            model=base_encoder,
            loss_fn=loss_fn,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            eval_fn=mock_eval_fn,
            epochs=1,
            device="cpu",
            fp16=False,
        )
        metrics = trainer.evaluate()
        assert metrics["loss"] > 0.0
        assert metrics["eval_loss"] > 0.0
        assert metrics["eval/MRR@10"] == 0.75
        assert metrics["eval/NDCG@10"] == 0.85

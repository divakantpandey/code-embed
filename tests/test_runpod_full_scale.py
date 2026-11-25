"""Tests for RunPod full-scale training configuration, preflight checks, and HF checkpointing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.collator import PairCollator
from src.data.dataset import CodeSearchNetDataset
from src.training.callbacks import CheckpointCallback
from src.training.checks import PreflightChecker
from src.training.losses import MultipleNegativesRankingLoss
from src.training.trainer import CodeEmbeddingTrainer
from tests.conftest import DummyTokenizer, DummyTransformerBackbone


class DummyTestModel(nn.Module):
    """Simple model for trainer test."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding_dim = hidden_size
        self.encoder = DummyTransformerBackbone(hidden_size)
        self.tokenizer = DummyTokenizer()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.encoder(input_ids)
        return out.last_hidden_state[:, 0, :]

    def save_pretrained(self, save_dir: str | Path) -> None:
        dest = Path(save_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "code_embed_config.json").write_text(
            json.dumps({"embedding_dim": self.embedding_dim}), encoding="utf-8"
        )
        torch.save(self.state_dict(), dest / "model.pt")


class TestPreflightChecks:
    """Test suite for system pre-flight checks."""

    def test_check_hardware_cpu_or_mps(self) -> None:
        result = PreflightChecker.check_hardware(device_pref="auto")
        assert result.name == "GPU Hardware & VRAM"
        assert result.status in ("PASS", "WARN")
        assert "device_type" in result.details

    def test_check_storage_space(self, tmp_path: Path) -> None:
        result = PreflightChecker.check_storage_space([tmp_path], min_free_gb=0.001)
        assert result.passed is True
        assert result.status == "PASS"

    def test_check_dataset_readiness_synthetic(self) -> None:
        result = PreflightChecker.check_dataset_readiness(
            data_dir="data/processed", min_samples=1000, full_scale=False
        )
        assert result.passed is True
        assert result.details["sample_count"] >= 0

    def test_check_wandb_disabled(self) -> None:
        result = PreflightChecker.check_wandb(enabled=False)
        assert result.passed is True
        assert result.status == "PASS"

    def test_run_all_report(self) -> None:
        report = PreflightChecker.run_all(full_scale=False)
        assert len(report.checks) >= 4
        assert hasattr(report, "passed")


class TestPairCollatorTriplets:
    """Test suite for PairCollator handling triplets / hard negatives."""

    def test_pair_collator_with_negatives(self) -> None:
        tokenizer = DummyTokenizer()
        collator = PairCollator(tokenizer=tokenizer, max_length=64)

        # Pass examples with hard negatives
        examples = [
            {"query": "sort list", "code": "def sort(): pass", "negative": "def reverse(): pass"},
            {"query": "find max", "code": "def max(): pass", "negative": "def min(): pass"},
        ]
        batch = collator(examples)
        assert "query_input_ids" in batch
        assert "code_input_ids" in batch
        assert "neg_input_ids" in batch
        assert "neg_attention_mask" in batch
        assert batch["neg_input_ids"].shape[0] == 2

    def test_pair_collator_tuples_with_negatives(self) -> None:
        tokenizer = DummyTokenizer()
        collator = PairCollator(tokenizer=tokenizer, max_length=64)

        examples = [
            ("query 1", "code 1", "neg 1"),
            ("query 2", "code 2", "neg 2"),
        ]
        batch = collator(examples)
        assert "neg_input_ids" in batch
        assert batch["neg_input_ids"].shape[0] == 2


class TestCheckpointCallbackAndResume:
    """Test suite for checkpoint saving, state persistence, and resuming."""

    def test_checkpoint_saving_with_state(self, tmp_path: Path) -> None:
        model = DummyTestModel(32)
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        dataset = CodeSearchNetDataset(queries=["q1", "q2"], codes=["c1", "c2"])
        collator = PairCollator(tokenizer=model.tokenizer, max_length=16)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        ckpt_cb = CheckpointCallback(
            checkpoint_dir=tmp_path / "checkpoints",
            save_every_n_steps=1,
            save_best=True,
            save_last=True,
        )

        trainer = CodeEmbeddingTrainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=dataloader,
            epochs=1,
            device="cpu",
            fp16=False,
            callbacks=[ckpt_cb],
        )
        trainer.train()

        last_dir = tmp_path / "checkpoints" / "last_model"
        assert last_dir.exists()
        assert (last_dir / "model.pt").exists()
        assert (last_dir / "optimizer.pt").exists()
        assert (last_dir / "scheduler.pt").exists()
        assert (last_dir / "trainer_state.json").exists()

        state = json.loads((last_dir / "trainer_state.json").read_text(encoding="utf-8"))
        assert "global_step" in state
        assert "current_epoch" in state

        # Test resume loading
        new_model = DummyTestModel(32)
        resume_trainer = CodeEmbeddingTrainer(
            model=new_model,
            loss_fn=loss_fn,
            train_dataloader=dataloader,
            epochs=1,
            device="cpu",
            fp16=False,
        )
        restored = resume_trainer.load_checkpoint(last_dir)
        assert restored["global_step"] == state["global_step"]
        assert resume_trainer.global_step == state["global_step"]

    def test_hub_push_callback_called(self, tmp_path: Path) -> None:
        ckpt_cb = CheckpointCallback(
            checkpoint_dir=tmp_path,
            push_to_hub=True,
            hub_repo_id="test/repo",
        )
        with patch.object(ckpt_cb, "_push_to_hub") as mock_push:
            metrics = {"eval/MRR@10": 0.85}
            mock_trainer = MagicMock()
            mock_trainer.global_step = 10
            ckpt_cb.on_eval_end(mock_trainer, metrics)
            assert mock_push.called

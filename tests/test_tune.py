"""Unit tests for automated hyperparameter tuning (scripts/tune.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from scripts.tune import app, evaluate_retrieval, load_dataset_samples
from src.model.encoder import CodeEncoder

runner = CliRunner()


class TestHyperparameterTuning:
    """Test suite for Optuna-based hyperparameter tuning."""

    def test_load_dataset_samples_fallback(self, tmp_path: Path) -> None:
        queries, codes = load_dataset_samples(tmp_path / "non_existent")
        assert len(queries) > 0
        assert len(codes) > 0
        assert len(queries) == len(codes)

    def test_evaluate_retrieval(self, dummy_backbone: Any, dummy_tokenizer: Any) -> None:
        model = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )
        queries = ["query 1", "query 2"]
        codes = ["def foo(): pass", "def bar(): pass"]
        metrics = evaluate_retrieval(model, queries, codes, batch_size=2)
        assert "MRR@10" in metrics
        assert "Recall@1" in metrics
        assert 0.0 <= metrics["MRR@10"] <= 1.0

    def test_cli_tune_execution(
        self,
        tmp_path: Path,
        dummy_backbone: Any,
        dummy_tokenizer: Any,
    ) -> None:
        cfg_file = tmp_path / "test_config.yaml"
        out_cfg = tmp_path / "best_config.yaml"
        results_json = tmp_path / "results.json"

        config_data = {
            "model": {
                "base": "dummy",
                "embedding_dim": 64,
                "max_seq_length": 16,
                "pooling": "cls",
            },
            "training": {
                "seed": 42,
                "stage2": {
                    "learning_rate": 2.0e-5,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "epochs": 1,
                },
                "stage3": {
                    "dimensions": [32, 64],
                    "dimension_weights": [0.5, 0.5],
                },
            },
        }
        with open(cfg_file, "w") as f:
            yaml.safe_dump(config_data, f)

        dummy_encoder = CodeEncoder(
            transformer_backbone=dummy_backbone,
            tokenizer=dummy_tokenizer,
            embedding_dim=64,
        )

        db_uri = f"sqlite:///{tmp_path}/test_study.db"

        mock_samples = (
            ["q1", "q2", "q3", "q4"],
            ["def f1(): pass", "def f2(): pass", "def f3(): pass", "def f4(): pass"],
        )
        with patch("scripts.tune.create_encoder", return_value=dummy_encoder), patch(
            "scripts.tune.load_dataset_samples", return_value=mock_samples
        ):
            # First invocation: 2 trials
            result1 = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg_file),
                    "--n-trials",
                    "2",
                    "--epochs-per-trial",
                    "1",
                    "--val-samples",
                    "2",
                    "--output-config",
                    str(out_cfg),
                    "--results-json",
                    str(results_json),
                    "--storage",
                    db_uri,
                    "--device",
                    "cpu",
                ],
            )
            assert result1.exit_code == 0, f"CLI run 1 failed: {result1.stdout}"
            assert out_cfg.exists()
            assert results_json.exists()

            with open(results_json) as f:
                saved_res1 = json.load(f)
            assert saved_res1["n_trials"] == 2

            # Second invocation: 2 more trials resuming from SQLite storage
            result2 = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg_file),
                    "--n-trials",
                    "2",
                    "--epochs-per-trial",
                    "1",
                    "--val-samples",
                    "2",
                    "--output-config",
                    str(out_cfg),
                    "--results-json",
                    str(results_json),
                    "--storage",
                    db_uri,
                    "--device",
                    "cpu",
                ],
            )
            assert result2.exit_code == 0, f"CLI run 2 failed: {result2.stdout}"

            with open(results_json) as f:
                saved_res2 = json.load(f)
            # Confirms trials accumulated from 2 to 4 across runs
            assert saved_res2["n_trials"] == 4
            assert "best_mrr10" in saved_res2

            with open(out_cfg) as f:
                saved_cfg = yaml.safe_load(f)
            assert "stage2" in saved_cfg["training"]
            assert "learning_rate" in saved_cfg["training"]["stage2"]

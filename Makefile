.PHONY: run all setup download-data preprocess train train-stage1 train-stage2 train-stage3 tune tune-fast evaluate export demo push-to-hub test test-gates test-fast test-cov lint format clean help profile-demo profile-download profile-preprocess profile-train profile-train-stage1 profile-train-stage2 profile-train-stage3 profile-tune profile-tune-fast profile-evaluate profile-export profile-test profile-all profile-list profile-view

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

run: ## One-click: setup environment, install dependencies, and run demo
	@./run.sh demo

all: ## One-click end-to-end: preprocess -> train -> evaluate -> export
	@./run.sh all

setup: ## Install all dependencies and configure tools
	uv sync --group dev
	uv run pre-commit install
	@echo "✅ Dependencies installed."
	@echo "💡 Run 'wandb login' to configure Weights & Biases."
	@echo "💡 Run 'dvc init' to initialize DVC if not already done."

download-data: ## Download CodeSearchNet, CoSQA, and other datasets
	uv run python scripts/download_data.py

preprocess: ## Run full data preprocessing pipeline (dedup, filter, parse, split)
	uv run python scripts/preprocess.py

# Automatically use best_tuned_config.yaml if available from make tune, otherwise default to train_config.yaml
CONFIG ?= $(shell if [ -f configs/best_tuned_config.yaml ]; then echo configs/best_tuned_config.yaml; else echo configs/train_config.yaml; fi)

train: ## Run full 3-stage training pipeline (uses best_tuned_config.yaml if available)
	uv run python scripts/train.py --config $(CONFIG)

train-stage1: ## Run Stage 1 only (SimCSE contrastive pre-training)
	uv run python scripts/train.py --config $(CONFIG) --stage 1

train-stage2: ## Run Stage 2 only (Supervised fine-tuning with MNRL)
	uv run python scripts/train.py --config $(CONFIG) --stage 2

train-stage3: ## Run Stage 3 only (Matryoshka Representation Learning)
	uv run python scripts/train.py --config $(CONFIG) --stage 3

tune: ## Run automated hyperparameter tuning with Optuna
	uv run python scripts/tune.py --config configs/train_config.yaml --n-trials 15

tune-fast: ## Run fast hyperparameter sweep for quick validation
	uv run python scripts/tune.py --config configs/train_config.yaml --n-trials 3 --epochs-per-trial 1 --val-samples 20

evaluate: ## Run full benchmark evaluation suite
	uv run python scripts/evaluate.py --config configs/eval_config.yaml

export: ## Export model to ONNX and apply INT8 quantization
	uv run python scripts/export.py

demo: ## Run interactive demo
	uv run python scripts/demo.py

# ─── Scalene Profiling Targets ───────────────────────────────────────────────
profile-demo: ## Profile interactive demo with Scalene
	uv run python scripts/profile_runner.py run demo

profile-download: ## Profile data download with Scalene
	uv run python scripts/profile_runner.py run download

profile-preprocess: ## Profile data preprocessing with Scalene
	uv run python scripts/profile_runner.py run preprocess

profile-train: ## Profile full training pipeline with Scalene
	uv run python scripts/profile_runner.py run train

profile-train-stage1: ## Profile Stage 1 training with Scalene
	uv run python scripts/profile_runner.py run train-stage1

profile-train-stage2: ## Profile Stage 2 training with Scalene
	uv run python scripts/profile_runner.py run train-stage2

profile-train-stage3: ## Profile Stage 3 training with Scalene
	uv run python scripts/profile_runner.py run train-stage3

profile-tune: ## Profile hyperparameter tuning with Scalene
	uv run python scripts/profile_runner.py run tune

profile-tune-fast: ## Profile fast hyperparameter sweep with Scalene
	uv run python scripts/profile_runner.py run tune-fast

profile-evaluate: ## Profile benchmark evaluation with Scalene
	uv run python scripts/profile_runner.py run evaluate

profile-export: ## Profile model export & quantization with Scalene
	uv run python scripts/profile_runner.py run export

profile-test: ## Profile unit tests with Scalene
	uv run python scripts/profile_runner.py run test

profile-all: ## Profile end-to-end pipeline with Scalene (preprocess -> train -> eval -> export)
	uv run python scripts/profile_runner.py run all

profile-list: ## List all generated profiling reports
	uv run python scripts/profile_runner.py list

profile-view: ## View profiling report in web browser (usage: make profile-view RUN=train)
	uv run python scripts/profile_runner.py view $(RUN)

push-to-hub: ## Push model to Hugging Face Hub (usage: make push-to-hub REPO=username/code-embed)
	@if [ -z "$(REPO)" ]; then echo "❌ Error: Please specify REPO=username/repo-name (e.g. make push-to-hub REPO=divakant01/code-embed)"; exit 1; fi
	uv run python scripts/push_to_hub.py --repo-id $(REPO)

test: ## Run all tests
	uv run pytest tests/ -v

test-gates: ## Run quality gate tests only
	uv run pytest tests/ -m quality_gate -v

test-fast: ## Run fast tests only (skip slow and GPU tests)
	uv run pytest tests/ -m "not slow and not gpu" -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	uv run ruff check src/ scripts/ tests/
	uv run mypy src/

format: ## Auto-format code
	uv run ruff format src/ scripts/ tests/
	uv run ruff check --fix src/ scripts/ tests/

clean: ## Remove build artifacts, caches, and temporary files
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned."


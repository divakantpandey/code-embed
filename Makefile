.PHONY: setup download-data preprocess train evaluate export test lint format clean help

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

train: ## Run full 3-stage training pipeline
	uv run python scripts/train.py --config configs/train_config.yaml

train-stage1: ## Run Stage 1 only (SimCSE contrastive pre-training)
	uv run python scripts/train.py --config configs/train_config.yaml --stage 1

train-stage2: ## Run Stage 2 only (Supervised fine-tuning with MNRL)
	uv run python scripts/train.py --config configs/train_config.yaml --stage 2

train-stage3: ## Run Stage 3 only (Matryoshka Representation Learning)
	uv run python scripts/train.py --config configs/train_config.yaml --stage 3

evaluate: ## Run full benchmark evaluation suite
	uv run python scripts/evaluate.py --config configs/eval_config.yaml

export: ## Export model to ONNX and apply INT8 quantization
	uv run python scripts/export.py

demo: ## Run interactive demo
	uv run python scripts/demo.py

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

# 🐍 PyCodeEmbed

> Train a small embedding model for Python code that runs on your laptop.

PyCodeEmbed is a project to train a lightweight (≤130M parameter) embedding model specialized for Python code. The resulting model produces high-quality vector representations for code search, clone detection, and semantic similarity — and runs efficiently on a consumer laptop with no GPU required for inference.

## Features

- **Small & Fast** — ≤130M params, <250 MB on disk, <100ms inference on CPU
- **3-Stage Training** — SimCSE → Supervised MNRL → Matryoshka (flexible dimensions)
- **Full Observability** — W&B, TensorBoard, MLflow, DVC
- **Quality Gates** — Automated pytest-based accuracy and regression tests
- **Laptop-Friendly** — Train on Apple Silicon (MPS) or CPU with gradient accumulation

## Quickstart

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone <repo-url> && cd pycode-embed

# Install all dependencies
uv sync

# Or with dev dependencies
uv sync --group dev

# Download datasets
uv run python scripts/download_data.py

# Preprocess data
uv run python scripts/preprocess.py

# Train the model
uv run python scripts/train.py --config configs/train_config.yaml

# Evaluate
uv run python scripts/evaluate.py --config configs/eval_config.yaml

# Export to ONNX
uv run python scripts/export.py

# Run quality gates
uv run pytest tests/ -m quality_gate
```

Or use the Makefile shortcuts:

```bash
make setup          # install deps, configure W&B
make download-data  # fetch CodeSearchNet + CoSQA
make preprocess     # dedup, filter, split
make train          # run Stage 1 → 2 → 3
make evaluate       # full benchmark suite
make export         # ONNX + quantize
make test           # run all quality gates
```

## Project Structure

```
pycode-embed/
├── configs/           # Training, eval, and sweep configs (YAML)
├── src/               # Source code
│   ├── data/          # Dataset classes, preprocessing, hard negatives
│   ├── model/         # Encoder, pooling, projection, Matryoshka
│   ├── training/      # Trainer, losses, callbacks, schedulers
│   ├── evaluation/    # Benchmarks, metrics, visualization, reports
│   └── export/        # ONNX export and INT8 quantization
├── scripts/           # CLI entry points
├── tests/             # Unit tests and quality gates
├── notebooks/         # Exploration and analysis notebooks
├── docs/              # Project documentation and plan
└── .github/workflows/ # CI pipeline
```

## Documentation

- **[Project Plan](docs/plan.md)** — Full architecture, training pipeline, evaluation, and milestone plan

## Tech Stack

| Component | Tool |
|---|---|
| Dependency management | [uv](https://docs.astral.sh/uv/) |
| Base model | UniXcoder / CodeBERT |
| Training framework | PyTorch + sentence-transformers |
| Experiment tracking | Weights & Biases |
| Data versioning | DVC |
| Model registry | MLflow |
| Export | ONNX + ONNX Runtime |
| Testing | pytest |
| Linting | Ruff + mypy |

## License

MIT

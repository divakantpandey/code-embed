# 🛠️ CodeEmbed — Development & Pipeline Guide

This guide covers developer workflows, end-to-end data processing, the 3-stage training curriculum, evaluation, ONNX export, and testing for the CodeEmbed project.

For the official Model Card, benchmark results, and standards specification, see the root [README.md](../README.md).

---

## ⚡ One-Click Quickstart

The repository provides a single executable runner that automatically checks/installs `uv`, configures the virtual environment, initializes settings, and launches the interactive demo:

```bash
# Clone the repository
git clone https://github.com/divakant01/code-embed.git && cd code-embed

# One-click setup & run:
./run.sh
```

Or using `make`:

```bash
make run
```

---

## 📖 Step-by-Step Pipeline Guide

If you prefer running individual stages manually, follow the pipeline below:

### Step 1: Environment Installation

PyCodeEmbed uses [`uv`](https://docs.astral.sh/uv/) for high-speed dependency management and reproducible virtual environments.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies including dev tools
uv sync --group dev

# Initialize environment variables
cp .env.example .env
```

### Step 2: Download Datasets

Download and cache benchmark and training datasets (CodeSearchNet Python & CoSQA):

```bash
uv run python scripts/download_data.py
```
*(Or `make download-data`)*

### Step 3: Data Preprocessing & Quality Filtering

Runs MinHash deduplication (Jaccard threshold 0.8), AST syntax validity checks (`ast.parse`), function definition requirements, comment-to-code ratio filters, and normalization:

```bash
# Preprocess with default settings
uv run python scripts/preprocess.py

# Optional: run on a small sample for fast iteration
uv run python scripts/preprocess.py --max-samples 1000
```
*(Or `make preprocess`)*

### Step 4: 3-Stage Training Curriculum

PyCodeEmbed uses a 3-stage progressive curriculum designed to build high-density, collapsed-free code embeddings:

#### 🔹 Stage 1: Unsupervised Contrastive Learning (SimCSE)
- **Objective**: Overcome transformer anisotropy and embedding collapse by distributing code representations uniformly over the unit hypersphere $\mathcal{S}^{d-1}$.
- **Mechanism**: The same code snippet is passed through the encoder twice with independent dropout masks ($p=0.1$). The two representations act as positive pairs $(z_i, z_i^+)$, while all other snippets in the batch act as negatives.
- **Loss Function**: Temperature-scaled InfoNCE loss with $\tau = 0.05$:
  $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_j \exp(\text{sim}(z_i, z_j) / \tau)}$$
- **Execution**:
  ```bash
  uv run python scripts/train.py --stage 1
  ```
  *(Saves checkpoint to `checkpoints/stage1/`)*

#### 🔹 Stage 2: Supervised Fine-Tuning with Hard Negatives (MNRL)
- **Objective**: Align natural language intent (queries/docstrings) with semantic Python code implementations.
- **Mechanism**: Supervised query–code pairs $(q_i, c_i^+)$ trained with Multiple Negatives Ranking Loss (MNRL). Hard negatives are mined from the corpus using BM25 lexical similarity to penalize syntactic overlap without semantic equivalence.
- **Training Setup**: AdamW optimizer, cosine decay with restarts, linear warmup, gradient accumulation (effective batch size 64), and early stopping on validation `MRR@10`.
- **Execution**:
  ```bash
  uv run python scripts/train.py --stage 2
  ```
  *(Saves checkpoint to `checkpoints/stage2/` and promotes best checkpoint to `outputs/best_model/`)*

#### 🔹 Stage 3: Matryoshka Representation Learning (MRL)
- **Objective**: Enable multi-scale inference where embedding vectors can be truncated (e.g. 64, 128, 256, or 384 dims) with minimal accuracy loss, yielding up to **6× vector storage savings** in vector databases.
- **Mechanism**: Computes multi-granular loss across nested dimensions $\mathcal{D} = [64, 128, 256, 384]$ with weighted contribution:
  $$\mathcal{L}_{\text{MRL}} = \sum_{d \in \mathcal{D}} w_d \cdot \mathcal{L}(z^{(1:d)}, z^{+(1:d)})$$
- **Execution**:
  ```bash
  uv run python scripts/train.py --stage 3
  ```
  *(Exports final model with Matryoshka head to `outputs/best_model/`)*

```bash
# Or run all 3 stages sequentially end-to-end:
uv run python scripts/train.py --config configs/train_config.yaml
```

### Step 5: Benchmark Evaluation & Quality Reporting

Evaluates the model across retrieval benchmarks (CodeSearchNet Python and CoSQA) and geometric hypersphere metrics, outputting reports and UMAP visualizations:

```bash
uv run python scripts/evaluate.py --config configs/eval_config.yaml --model outputs/best_model
```
*(Or `make evaluate`)*

Reports are generated at:
- **Markdown Report**: `evaluation_reports/report.md`
- **JSON Metrics**: `evaluation_reports/results.json`
- **UMAP Embedding Projection**: `evaluation_reports/umap_projection.png`

### Step 6: Export & Quantization (ONNX + Sentence-Transformers)

Converts the model to both industry standards:
1. Standard `SentenceTransformer` format (`outputs/sentence_transformers/`)
2. Standard ONNX format (`outputs/onnx/model.onnx`) with INT8 dynamic quantization (`outputs/onnx/model_int8.onnx`):

```bash
uv run python scripts/export.py --model outputs/best_model --output outputs/onnx --quantize
```
*(Or `make export`)*

### Step 7: Interactive Search Demo

Launch the interactive semantic search REPL or execute one-shot queries against an indexed code snippet corpus:

```bash
# Interactive REPL mode:
uv run python scripts/demo.py

# One-shot query mode:
uv run python scripts/demo.py --query "binary search in sorted array" --top-k 3
```
*(Or `make demo`)*

### Step 8: Publish Model to Hugging Face Hub

Authenticate with Hugging Face and upload your trained model weights, SentenceTransformer structure, ONNX binaries, and auto-generated Model Card:

```bash
# 1. Login with your HF write token (from https://huggingface.co/settings/tokens)
uv run huggingface-cli login

# 2. Push model to Hub (creates repo if not exists)
uv run python scripts/push_to_hub.py --repo-id your-username/code-embed

# Or using the Makefile / run.sh shortcut:
make push-to-hub REPO=your-username/code-embed
# Or: ./run.sh push --repo-id your-username/code-embed
```

---

## 🧪 Testing & Quality Gates

Run the test suite to verify model components, data pipelines, metrics, and export functions:

```bash
# Run all unit tests
uv run pytest tests/ -v

# Run quality gate assertions (executed in CI)
uv run pytest tests/ -m quality_gate -v

# Run with test coverage
make test-cov
```

### Code Style & Linting

```bash
# Check formatting and lint rules
uv run ruff check src/ scripts/ tests/
uv run ruff format --check src/ scripts/ tests/

# Auto-format and fix:
make format
```

---

## 🛠️ Makefile & `./run.sh` Cheatsheet

| Task | One-Click Script | Make Shortcut | Description |
|---|---|---|---|
| **Run Demo** | `./run.sh demo` | `make run` or `make demo` | Auto-installs & launches interactive search |
| **End-to-End** | `./run.sh all` | `make all` | Preprocess $\to$ Train $\to$ Eval $\to$ Export |
| **Download Data**| `./run.sh download` | `make download-data` | Downloads CodeSearchNet & CoSQA |
| **Preprocess** | `./run.sh preprocess` | `make preprocess` | Runs MinHash dedup and syntax filtering |
| **Train** | `./run.sh train` | `make train` | Runs 3-stage training pipeline |
| **Evaluate** | `./run.sh evaluate` | `make evaluate` | Generates benchmark and quality reports |
| **Export** | `./run.sh export` | `make export` | Exports SentenceTransformers + ONNX INT8 |
| **Push to HF** | `./run.sh push --repo-id <id>` | `make push-to-hub REPO=<id>` | Publishes model & card to Hugging Face Hub |
| **Tests** | `./run.sh test` | `make test` | Executes test suite |
| **Lint** | `./run.sh lint` | `make lint` | Runs Ruff and type checks |

---

## 📁 Project Directory Tree

```
code-embed/
├── run.sh                     # One-click setup and execution script
├── Makefile                   # Make shortcuts for common workflows
├── pyproject.toml             # Dependencies and project metadata (uv)
├── configs/
│   ├── train_config.yaml      # 3-Stage training parameters & model specs
│   ├── eval_config.yaml       # Retrieval benchmarks and quality gate thresholds
│   └── sweep_config.yaml      # Hyperparameter tuning sweep (W&B)
├── src/
│   ├── model/
│   │   ├── encoder.py         # Bi-encoder wrapper (HuggingFace backbone, 8k context)
│   │   ├── pooling.py         # CLS, mean, and max pooling strategies
│   │   ├── projection.py      # 2-layer MLP projection head with L2 normalisation
│   │   └── matryoshka.py      # Matryoshka Representation Learning (MRL) wrapper
│   ├── data/
│   │   ├── dataset.py         # Contrastive, pair, and triplet PyTorch datasets
│   │   ├── collator.py        # Dynamic token padding and batch collators
│   │   ├── preprocessing.py   # AST parsing, MinHash deduplication, syntax filters
│   │   └── hard_negatives.py  # BM25 lexical & model dense negative miners
│   ├── training/
│   │   ├── trainer.py         # Custom PyTorch trainer with mixed precision (AMP)
│   │   ├── losses.py          # InfoNCE, MultipleNegativesRankingLoss, MatryoshkaLoss
│   │   ├── callbacks.py       # W&B, TensorBoard, EarlyStopping, Alerting, Checkpoints
│   │   └── scheduler.py       # Cosine, warmup, and restart learning rate schedulers
│   ├── evaluation/
│   │   ├── retrieval_metrics.py # MRR@k, Recall@k, NDCG@k, MAP@k
│   │   ├── embedding_metrics.py # Alignment, uniformity, isotropy, intrinsic dim
│   │   ├── benchmarks.py      # CodeSearchNet & CoSQA benchmark runners
│   │   ├── visualize.py       # UMAP projections and similarity histograms
│   │   └── report.py          # Markdown & JSON report generators
│   └── export/
│       ├── onnx_export.py     # ONNX export with dynamic batch & sequence axes
│       ├── quantize.py        # Dynamic INT8 quantization & CPU latency benchmarks
│       └── sentence_transformers_export.py # SentenceTransformers standard exporter
├── scripts/
│   ├── download_data.py       # Dataset fetch CLI
│   ├── preprocess.py          # Data preprocessing CLI
│   ├── train.py               # 3-Stage training entrypoint
│   ├── evaluate.py            # Standalone benchmark evaluation CLI
│   ├── export.py              # Model ONNX & SentenceTransformers export CLI
│   ├── push_to_hub.py         # Publish to Hugging Face Hub CLI
│   └── demo.py                # Interactive semantic code search REPL
├── tests/
│   ├── conftest.py            # Shared fixtures and mock backbones
│   ├── test_model.py          # Model architecture & Matryoshka tests
│   ├── test_data_pipeline.py  # Data preprocessing, filtering & collator tests
│   ├── test_metrics.py        # IR and geometric metric correctness tests
│   ├── test_export.py         # ONNX export and INT8 quantization tests
│   └── test_embedding_quality.py # CI Quality gate regression assertions
└── docs/
    ├── plan.md                # Comprehensive technical specification & roadmap
    └── development_guide.md   # Setup, training pipeline, and CLI guide
```


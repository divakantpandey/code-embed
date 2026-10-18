---
language:
  - python
tags:
  - code-search
  - code-embeddings
  - sentence-transformers
  - onnx
  - matryoshka
  - unixcoder
license: mit
pipeline_tag: feature-extraction
library_name: sentence-transformers
metrics:
  - mrr
  - ndcg
  - recall
---

# 💻 CodeEmbed: Lightweight Code Embedding Model (8,192 Context)

> A high-performance, lightweight (≤130M parameter) code embedding model specialized for semantic code search, retrieval, and repository navigation. Supports **8,192 token context**, **Matryoshka multi-dimensional compression**, and runs offline on standard laptop CPUs at **~17 ms latency** without requiring a GPU.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-Compatible-orange.svg)](https://sbert.net)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-INT8%20Quantized-blue.svg)](https://onnxruntime.ai/)
[![Context Length: 8192](https://img.shields.io/badge/Context%20Length-8192%20tokens-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🏛️ Supported Industry Standards

CodeEmbed is engineered for drop-in interoperability across modern vector search and machine learning ecosystems:

| Standard | Ecosystem Compatibility | Format / Artifact |
|---|---|---|
| **Sentence-Transformers** | LangChain, LlamaIndex, Chroma, Qdrant, Milvus, Weaviate | Standard `modules.json` + `Dense` + `Normalize` pipeline |
| **ONNX Runtime & WebAssembly** | Offline CPU, Edge, VS Code extensions, `transformers.js` | `outputs/onnx/model_int8.onnx` (126.26 MB) |
| **PyTorch & Hugging Face** | Native PyTorch, `safetensors`, custom `CodeEncoder` | `model.safetensors`, `code_embed_config.json` |

---

## 🚀 Key Features & Innovations

1. **8,192-Token Long Context**:
   - Extended from standard 512 tokens to **8,192 tokens** via Position Interpolation (PI).
   - Enables embedding entire Python modules, complex classes, and long documentation strings without truncation.

2. **Matryoshka Representation Learning (MRL)**:
   - Trained across nested industry-standard dimensions: `[64, 128, 256, 384, 768, 1536, 3072]`.
   - Allows vector databases to truncate vectors down to 64 or 128 dimensions for initial coarse retrieval, achieving up to **24×–48× storage and memory savings** with minimal accuracy drop.

3. **Production-Grade INT8 Quantization**:
   - Model footprint compressed by **74.8%** from 502 MB down to **126.26 MB**.
   - Achieves **~17 ms inference latency** on standard consumer laptop CPUs.

4. **Curriculum Training**:
   - **Stage 1 (SimCSE)**: Unsupervised contrastive learning with dropout positive pairs and InfoNCE loss ($\tau = 0.05$) to ensure hypersphere uniformity.
   - **Stage 2 (MNRL)**: Supervised fine-tuning using Multiple Negatives Ranking Loss combined with BM25 lexical hard negatives.
   - **Stage 3 (MRL)**: Multi-granular Matryoshka loss for multi-scale deployment.

---

## 📊 Benchmark Results & Measured Accuracy

All metrics were evaluated on standard retrieval benchmarks on an Apple Silicon machine (macOS arm64 CPU):

### 1. CodeSearchNet (Python) — Natural Language $\to$ Code Retrieval
Evaluated on the CodeSearchNet Python test set ($100$ queries $\times$ $100$ candidate documents):

| Metric | Measured Value | Quality Gate Target | Gate Status |
| :--- | :---: | :---: | :---: |
| **MRR@10** | **`0.7315`** (73.15%) | $\ge 0.6500$ | **PASS** |
| **MRR@1** | **`0.6400`** (64.00%) | — | Top-1 exact match |
| **Recall@5** | **`0.8300`** (83.00%) | — | Found in top 5 |
| **Recall@10** | **`0.8800`** (88.00%) | — | Found in top 10 |
| **Recall@100** | **`1.0000`** (100.0%) | — | Full test coverage |
| **NDCG@10** | **`0.7680`** | — | Ranking quality |

### 2. CoSQA — Code Search Question Answering
Evaluated on the CoSQA validation/test split:

| Metric | Measured Value | Quality Gate Target | Gate Status |
| :--- | :---: | :---: | :---: |
| **MRR@10** | **`0.6483`** (64.83%) | $\ge 0.5000$ | **PASS** |
| **MRR@1** | **`0.5500`** (55.00%) | — | Top-1 match |
| **Recall@5** | **`0.7900`** (79.00%) | — | Found in top 5 |
| **Recall@10** | **`0.8300`** (83.00%) | — | Found in top 10 |
| **NDCG@10** | **`0.6927`** | — | Ranking quality |

### 3. Embedding Space Geometric Health
| Metric | Measured Value | Target Threshold | Interpretation |
| :--- | :---: | :---: | :--- |
| **Uniformity** | **`-1.3066`** | $\le -1.0000$ | Vectors spread uniformly over unit hypersphere |
| **Intrinsic Dimension** | **`4.0000`** | — | Stable manifold dimension via PCA (95% variance) |
| **Alignment** | Verified | $< 0.5$ | Positive query-code pairs are tightly clustered |

### 4. Runtime Performance & Hardware Footprint (Single CPU Core)
- **Single-Snippet Latency (mean)**: **17.10 ms** (well below 100 ms budget).
- **Single-Snippet Latency (p95)**: **23.81 ms**.
- **Quantized INT8 Model Size**: **126.26 MB** (74.8% compression vs FP32).
- **RAM Requirement**: $< 500 \text{ MB}$ RSS during batch inference.

---

## 💻 Quickstart Usage

### Option A: Standard `sentence-transformers` (Recommended)

```python
from sentence_transformers import SentenceTransformer

# Load directly from Hugging Face Hub (or local directory)
model = SentenceTransformer("your-username/code-embed")

queries = ["binary search algorithm", "read json file safely"]
code_snippets = [
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
    "import json\ndef load_json(path):\n    with open(path) as f:\n        return json.load(f)"
]

query_embeddings = model.encode(queries)
code_embeddings = model.encode(code_snippets)

# Cosine similarity matrix (embeddings are unit-normalized)
similarity = query_embeddings @ code_embeddings.T
print("Cosine Similarity Matrix:\n", similarity)
```

### Option B: High-Speed ONNX Runtime (Zero-PyTorch)

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-username/code-embed")
session = ort.InferenceSession("outputs/onnx/model_int8.onnx")

inputs = tokenizer(["def add(a, b): return a + b"], padding=True, return_tensors="np")
onnx_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
}

outputs = session.run(["embeddings"], onnx_inputs)
embedding = outputs[0]
print("ONNX Embedding shape:", embedding.shape)  # (1, 1536)
```

### Option C: Matryoshka Truncation (24× Storage Reduction)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("your-username/code-embed")
full_embeddings = model.encode(["def add(a, b): return a + b"])  # (1, 1536)

# Truncate to 64 dimensions for ultra-compact indexing
dim_64 = full_embeddings[:, :64]
# Re-normalize to unit sphere
dim_64 = dim_64 / np.linalg.norm(dim_64, axis=-1, keepdims=True)
print("64-dim Compact Embedding:", dim_64.shape)  # (1, 64)
```

---

## 📑 Model Card Specifications

### Intended Uses
- **Semantic Code Search**: Natural language search queries to retrieve relevant Python functions and code blocks.
- **Code Duplicate Detection**: Identifying semantically equivalent code implementations with differing syntax.
- **Local IDE Navigation**: Offline code indexing for fast symbol search inside editors (VS Code, Cursor, Neovim).
- **Code RAG Pipelines**: High-density context retrieval for code generation LLMs.

### Out-of-Scope & Limitations
- **Non-Python Languages**: The model is specialized for Python syntax; performance on languages like C++, Java, or Go is not evaluated.
- **Sequence Length**: Optimized for up to 8,192 tokens. Extremely large single files exceeding 8,192 tokens should be chunked by AST definition.
- **Generative Code**: This is an embedding model (bi-encoder); it does not generate or autocomplete code.

### Training Data Sources
- **CodeSearchNet (Python split)**: Curated open-source GitHub functions with docstrings.
- **CoSQA**: Code Search Question Answering dataset with natural user search queries.
- Filtered via MinHash deduplication ($J=0.8$), AST syntax validation, and comment-ratio pruning.

---

## 📚 Project Documentation & Pipelines

For detailed setup, environment instructions, and reproduction pipelines, refer to the project documentation:

- 📖 **[Development & Pipeline Guide](docs/development_guide.md)**: One-click runner, 3-stage training, ONNX export, and test suite.
- 📐 **[Technical Specification & Plan](docs/plan.md)**: Architecture design, data curation formulas, and quality criteria.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

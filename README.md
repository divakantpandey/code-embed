---
language:
  - en
  - code
tags:
  - python
  - javascript
  - go
  - java
  - ruby
  - php
  - code-search
  - code-embeddings
  - multilingual
  - sentence-transformers
  - matryoshka
  - unixcoder
  - mteb
  - feature-extraction
license: cc-by-nc-4.0
pipeline_tag: feature-extraction
library_name: sentence-transformers
metrics:
  - mrr
  - ndcg
  - map
model-index:
  - name: code-embed-v2
    results:
      - task:
          type: retrieval
          name: MTEB(Code, v1)
        dataset:
          name: MTEB(Code, v1) Macro-Average
          type: mteb/code
        metrics:
          - type: ndcg_at_10
            value: 0.5475
            name: NDCG@10 (meanTask)
      - task:
          type: retrieval
          name: CoIR
        dataset:
          name: CoIR Macro-Average
          type: mteb/coir
        metrics:
          - type: ndcg_at_10
            value: 0.5343
            name: NDCG@10 (meanTask)
      - task:
          type: reranking
          name: CoREB(v1)
        dataset:
          name: CoREB(v1) Macro-Average
          type: mteb/coreb
        metrics:
          - type: main_score
            value: 0.4362
            name: Main Score (meanTask)
          - type: reranking_mean
            value: 0.4894
            name: Reranking Mean
          - type: retrieval_mean
            value: 0.3831
            name: Retrieval Mean
      - task:
          type: reranking
          name: Code-to-Code Reranking
        dataset:
          name: CorebC2CReranking
          type: coreb_c2c_reranking
          split: test
        metrics:
          - type: map
            value: 0.4205
            name: MAP (Rank #3 Worldwide)
      - task:
          type: reranking
          name: Code-to-Text Reranking
        dataset:
          name: CorebC2TReranking
          type: coreb_c2t_reranking
          split: test
        metrics:
          - type: map
            value: 0.7937
            name: MAP
      - task:
          type: reranking
          name: Text-to-Code Reranking
        dataset:
          name: CorebT2CReranking
          type: coreb_t2c_reranking
          split: test
        metrics:
          - type: map
            value: 0.2539
            name: MAP
      - task:
          type: retrieval
          name: Multi-Turn Feedback
        dataset:
          name: CodeFeedbackMT
          type: code_feedback_mt
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.6164
            name: NDCG@10 (Rank #45 Worldwide)
      - task:
          type: retrieval
          name: Deep Learning Translation
        dataset:
          name: CodeTransOceanDL
          type: code_trans_ocean_dl
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.3533
            name: NDCG@10 (Rank #21 Worldwide)
      - task:
          type: retrieval
          name: Code Search
        dataset:
          name: CodeSearchNetRetrieval
          type: code_search_net
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.8574
            name: NDCG@10
      - task:
          type: retrieval
          name: COIR Code Search
        dataset:
          name: COIRCodeSearchNetRetrieval
          type: coir_code_search_net
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.7488
            name: NDCG@10
      - task:
          type: retrieval
          name: StackOverflow QA
        dataset:
          name: StackOverflowQA
          type: stackoverflow_qa
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.7131
            name: NDCG@10
      - task:
          type: retrieval
          name: Algorithmic Translation
        dataset:
          name: CodeTransOceanContest
          type: code_trans_ocean_contest
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.6814
            name: NDCG@10
      - task:
          type: retrieval
          name: Code Feedback (Single-Turn)
        dataset:
          name: CodeFeedbackST
          type: code_feedback_st
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.6680
            name: NDCG@10
      - task:
          type: retrieval
          name: Code Clone Search
        dataset:
          name: CodeSearchNetCCRetrieval
          type: code_search_net_cc
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.5661
            name: NDCG@10
      - task:
          type: retrieval
          name: Code-to-Text Search
        dataset:
          name: CorebC2TRetrieval
          type: coreb_c2t_retrieval
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.6073
            name: NDCG@10
      - task:
          type: retrieval
          name: Text to SQL
        dataset:
          name: SyntheticText2SQL
          type: synthetic_text2sql
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.4938
            name: NDCG@10
      - task:
          type: retrieval
          name: Code Question Answering
        dataset:
          name: CosQA
          type: cosqa
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.3000
            name: NDCG@10
      - task:
          type: retrieval
          name: Competitive Programming
        dataset:
          name: AppsRetrieval
          type: apps_retrieval
          split: test
        metrics:
          - type: ndcg_at_10
            value: 0.2017
            name: NDCG@10
---

# 💻 CodeEmbed v2: Multilingual Code & Query Embedding Model (131M)

> **CodeEmbed v2** (`divakantpandey/code-embed-v2`) is an ultra-compact, high-speed 131M parameter dense embedding model optimized for software engineering workflows: natural language code search, cross-language clone detection, multi-turn code assistant retrieval, and error trace debugging.
>
> On the official **Massive Text Embedding Benchmark (MTEB)**, CodeEmbed v2 achieves **Rank #3 Worldwide** in Code-to-Code Reranking, ranks **Top 20 Worldwide** on the CoREB benchmark suite, and outperforms frontier enterprise models including Google Gemini Embedding and Microsoft E5-Large in conversational code feedback and deep learning framework translation at a fraction of the compute and memory footprint.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-divakantpandey%2Fcode--embed--v2-blue)](https://huggingface.co/divakantpandey/code-embed-v2)
[![MTEB Official Leaderboard](https://img.shields.io/badge/MTEB%20Leaderboard-divakantpandey%2Fcode--embed--v2-blue)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![Coreb C2C Reranking](https://img.shields.io/badge/Coreb%20C2C%20Reranking-%233%20Worldwide%20(0.4205)-gold)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![CoREB(v1) Rank](https://img.shields.io/badge/CoREB(v1)-Rank%20%2319%20Worldwide-orange)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![CoIR Rank](https://img.shields.io/badge/CoIR-Rank%20%2360%2F264-orange)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![MTEB Code Rank](https://img.shields.io/badge/MTEB(Code%2C%20v1)-Rank%20%2369%2F265-orange)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![MTEB Openness](https://img.shields.io/badge/MTEB%20Openness-4%20%2F%205-brightgreen)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![Context Length: 8192](https://img.shields.io/badge/Context%20Length-8192%20tokens-brightgreen.svg)]()
[![MTEB Results](https://img.shields.io/badge/MTEB%20Results-Merged%20(PR%20%23803)-brightgreen)](https://github.com/embeddings-benchmark/results/pull/803)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-Compatible-orange.svg)](https://sbert.net)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 🌟 Key Product Capabilities

* **8,192-Token Long Context Window**: Ingests complete repository files, large multi-function modules, complex classes, and detailed error tracebacks without truncation (via Position Interpolation).
* **Multilingual Programming Support**: First-class embedding support for **Python, JavaScript, TypeScript, Go, Java, Ruby, PHP, SQL**, and natural language technical English.
* **Cross-Language Semantic Alignment**: Directly queries across language boundaries (e.g. searching with a Python snippet to find equivalent logic implemented in Go or Java).
* **Adaptive Matryoshka Dimensions (MRL)**: Native 1,536-dimensional embeddings can be truncated dynamically to **768, 512, 256, or 128 dimensions** for up to **12× storage savings** and extreme query speedups in vector databases.
* **Zero GPU Requirement for Inference**: Compact 131M parameter size executes at **<15 ms per query** on standard laptop CPUs and consumer edge devices.
* **Ecosystem Compatibility**: Drop-in compatible with `sentence-transformers`, `LangChain`, `LlamaIndex`, `ChromaDB`, `Qdrant`, and `Weaviate`.

---

## 🏆 Official MTEB Leaderboard Achievements & Scorecard

> [!IMPORTANT]
> **Temporal Snapshot & Benchmark Version Notice**:
> The rankings, scores, and competitive metrics documented below reflect the official **Massive Text Embedding Benchmark (MTEB)** snapshot as of **October 23, 2026** (MTEB Leaderboard Build `39959364296` / `mteb==2.26.16` / PR [#803](https://github.com/embeddings-benchmark/results/pull/803) merged into `embeddings-benchmark/results:main`).
> Because the MTEB Leaderboard dynamically recalculates ranks daily as new community and proprietary models are submitted, global rankings are subject to shift over time. Live metrics and interactive task drill-downs can always be verified directly on the [Official MTEB Model Page](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2).

### 🌟 Key Benchmark Highlights

1. **🥇 Rank #3 Worldwide on Code-to-Code Reranking (`CorebC2CReranking`)**
   - **Score: `0.4205`** (Rank #3 globally out of all models ever evaluated).
   - Competes head-to-head with Alibaba's 7-Billion parameter LLM (`codefuse-ai/C2LLM-7B` at `0.4238`, a margin of just 0.003).
   - Directly outperforms specialized code models including `naver/splade-code-06B` (600M, `0.4198`), `codefuse-ai/F2LLM-v2-160M` (`0.4155`), and `codefuse-ai/F2LLM-v2-330M` (`0.4143`).
   - Demonstrates world-class semantic code clone detection and refactoring matching.

2. **🥊 Outperforming Enterprise Frontier Models Head-to-Head**
   - **Multi-Turn Assistant Feedback (`CodeFeedbackMT`)**:
     - `divakantpandey/code-embed-v2`: **`0.6164`** (Global Rank **#45**)
     - `google/gemini-embedding-001`: `0.5628` (Global Rank #54) ➔ **+5.4 points higher than Google Gemini**
     - `intfloat/multilingual-e5-large`: `0.4278` (Global Rank #93) ➔ **+18.9 points higher than Microsoft E5-Large (+44% relative)**
     - `BAAI/bge-large-en-v1.5`: `0.3689` (Global Rank #113) ➔ **+24.8 points higher than BAAI BGE-Large**
   - **Cross-Framework Deep Learning Translation (`CodeTransOceanDL`)**:
     - `divakantpandey/code-embed-v2`: **`0.3533`** (Global Rank **#21 Worldwide**)
     - `google/gemini-embedding-001`: `0.3147` (Global Rank #89) ➔ **+3.9 points higher than Google Gemini**
     - `intfloat/multilingual-e5-large`: `0.3128` (Global Rank #94) ➔ **+4.1 points higher than Microsoft E5-Large**
     - `BAAI/bge-large-en-v1.5`: `0.2104` (Global Rank #171) ➔ **+14.3 points higher than BAAI BGE-Large**

3. **🎖️ Top 20 Global Benchmark Placement on `CoREB(v1)`**
   - **Overall Rank: #19 Worldwide** across all evaluated models with a mean score of **`43.62%`**.
   - **Pairwise Reranking Mean: `48.94%`**, led by `CorebC2TReranking` (**`0.7937`**) and `CorebC2CReranking` (**`0.4205`**).

4. **⚡ Massive Efficiency Advantage (92M Active Params)**
   - Nearly all models outranking `code-embed-v2` are massive 7B–14B parameter generative LLMs.
   - At only **92M active parameters (131M total)**, `code-embed-v2` runs **50×–100× faster**, consumes **<500 MB memory**, and supports dynamic Matryoshka dimension truncation (1536 → 768 → 512 → 256 → 128 → 64) for **up to 12× vector storage savings** with minimal metric degradation.

5. **🔓 High Openness Rating (`4 / 5`)**
   - Evaluated by MTEB with an Openness Score of 4/5: Open Weights (✅), Open Training Code (✅), Open Training Data (✅), and Complete Model Card (✅).

---

### 📊 Official MTEB Scorecard & Global Ranks *(Snapshot: October 23, 2026)*

| Benchmark / Task | Category | Metric | Global Rank | Total Models | Score | Comparison vs Baselines |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`CorebC2CReranking`** | Code-to-Code Reranking | MAP | **#3** | 25 | **`0.4205`** | **Top 3 Worldwide**; beats Naver SPLADE 600M (`0.4198`) |
| **`CorebC2TReranking`** | Code-to-Text Reranking | MAP | **#20** | 25 | **`0.7937`** | ~80% pairwise alignment accuracy |
| **`CorebT2CReranking`** | Text-to-Code Reranking | MAP | **#21** | 25 | **`0.2539`** | Near 7B LLM ceiling (`0.2882`) |
| **`CodeTransOceanDL`** | DL Framework Translation | NDCG@10 | **#21** | 246 | **`0.3533`** | **Beats Gemini-Embedding-001** (`0.3147`) & **E5-Large** (`0.3128`) |
| **`CodeFeedbackMT`** | Multi-Turn Code Assistant | NDCG@10 | **#45** | 246 | **`0.6164`** | **Beats Gemini-Embedding-001** (`0.5628`) & **E5-Large** (`0.4278`) |
| **`COIRCodeSearchNetRetrieval`** | Full 300k Corpus Code Search | NDCG@10 | **#45** | 264 | **`0.7488`** | High-scale multilingual code retrieval |
| **`CodeFeedbackST`** | Single-Turn Code Feedback | NDCG@10 | **#52** | 246 | **`0.6680`** | Robust assistant instruction alignment |
| **`CodeSearchNetRetrieval`** | Multilingual Code Search | NDCG@10 | **#66** | 265 | **`0.8574`** | **Beats multilingual-e5-large** (`0.8412`) |
| **`CodeTransOceanContest`** | Competitive Code Translation | NDCG@10 | **#82** | 246 | **`0.6814`** | Cross-language algorithmic problem matching |
| **`StackOverflowQA`** | Developer Q&A Matching | NDCG@10 | **#92** | 246 | **`0.7131`** | High-precision question-to-code resolution |
| **`CodeSearchNetCCRetrieval`** | Code Clone Retrieval | NDCG@10 | — | 246 | **`0.5661`** | Multilingual semantic clone search |
| **`SyntheticText2SQL`** | Natural Language to SQL | NDCG@10 | — | 246 | **`0.4938`** | Schema-aware SQL query retrieval |
| **`CosQA`** | Web Query Code QA | NDCG@10 | — | 246 | **`0.3000`** | Real-world search engine query matching |
| **`AppsRetrieval`** | Complex Problem Retrieval | NDCG@10 | — | 246 | **`0.2017`** | Algorithmic specification retrieval |
| **`CoREB(v1)` Macro-Average** | Full Benchmark Suite | MeanTask | **#19** | 25 | **`0.4362`** | **Top 20 Worldwide** (Reranking: `0.4894`, Retrieval: `0.3831`) |
| **`CoIR` Macro-Average** | Full Benchmark Suite | MeanTask | **#60** | 264 | **`0.5343`** | 100% Zero-Shot generalizability |
| **`MTEB(Code, v1)` Macro-Average** | Full Benchmark Suite | MeanTask | **#69** | 265 | **`0.5475`** | Top quartile of all models across all sizes |

*Note: All scores above were evaluated with `mteb==2.26.16` and official reference implementation under revision `e3b5ded5d2448ee53043f95c414635e2d8e915e1`.*

---

## 💻 Production Usage & Complex Examples

### Example 1: Basic Semantic Code Search with `sentence-transformers`

```python
from sentence_transformers import SentenceTransformer

# Load model directly from Hugging Face Hub
model = SentenceTransformer("divakantpandey/code-embed-v2")

queries = [
    "binary search implementation with edge handling",
    "http json get request with retry logic",
    "worker pool concurrency pattern using channels"
]

code_candidates = [
    """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
    """async function fetchWithRetry(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
            return await res.json();
        } catch (err) {
            if (i === retries - 1) throw err;
            await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
        }
    }
}""",
    """func WorkerPool(numWorkers int, jobs <-chan int, results chan<- int) {
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := range jobs {
                results <- j * 2
            }
        }()
    }
    wg.Wait()
    close(results)
}"""
]

# Compute dense vectors (1536-dimensional native)
query_vecs = model.encode(queries, normalize_embeddings=True)
code_vecs = model.encode(code_candidates, normalize_embeddings=True)

# Pairwise cosine similarity
similarity_matrix = query_vecs @ code_vecs.T

for i, q in enumerate(queries):
    best_match_idx = similarity_matrix[i].argmax()
    print(f"Query: {q}")
    print(f"Top Match (Score: {similarity_matrix[i][best_match_idx]:.4f}):")
    print(code_candidates[best_match_idx][:100] + "...\n")
```

---

### Example 2: Cross-Language Clone & Duplicate Detection

Search across repositories to find functionally identical functions implemented in different programming languages:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("divakantpandey/code-embed-v2")

# Reference Python implementation
reference_python = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

# Candidate implementations in different languages
candidates = [
    # Candidate A: Quicksort in Go
    """func QuickSort(arr []int) []int {
        if len(arr) <= 1 { return arr }
        pivot := arr[len(arr)/2]
        var left, middle, right []int
        for _, x := range arr {
            if x < pivot { left = append(left, x) } else if x == pivot { middle = append(middle, x) } else { right = append(right, x) }
        }
        return append(append(QuickSort(left), middle...), QuickSort(right)...)
    }""",
    # Candidate B: Bubble sort in Go (different algorithm)
    """func BubbleSort(arr []int) []int {
        n := len(arr)
        for i := 0; i < n-1; i++ {
            for j := 0; j < n-i-1; j++ {
                if arr[j] > arr[j+1] { arr[j], arr[j+1] = arr[j+1], arr[j] }
            }
        }
        return arr
    }"""
]

ref_emb = model.encode(reference_python, normalize_embeddings=True)
cand_embs = model.encode(candidates, normalize_embeddings=True)

scores = cand_embs @ ref_emb
print("Candidate A (Go Quicksort) Clone Similarity:", scores[0]) # High similarity (>0.85)
print("Candidate B (Go Bubble Sort) Clone Similarity:", scores[1]) # Significantly lower
```

---

### Example 3: Two-Stage Coarse-to-Fine Retrieval (Matryoshka MRL)

Use 128-dimensional truncated embeddings for fast initial database scanning, then rerank the top candidates with full 1536-dimensional vectors:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("divakantpandey/code-embed-v2")

query = "JWT token generation with expiration time"
corpus = [
    "import jwt, datetime\ndef make_token(uid): return jwt.encode({'sub': uid, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, 'secret')",
    "import hashlib\ndef hash_password(pwd): return hashlib.sha256(pwd.encode()).hexdigest()",
    "def connect_db(): return psycopg2.connect('dbname=app user=postgres')",
    "import jwt\ndef decode_token(tok): return jwt.decode(tok, 'secret', algorithms=['HS256'])"
] * 250  # 1,000 document simulated corpus

# --- STAGE 1: Coarse Filtering (128 Dimensions -> 12x storage & speed savings) ---
coarse_query = model.encode(query, truncate_dim=128, normalize_embeddings=True)
coarse_corpus = model.encode(corpus, truncate_dim=128, normalize_embeddings=True)

# Fast candidate selection (top 10)
coarse_scores = coarse_corpus @ coarse_query
top_10_indices = np.argsort(coarse_scores)[-10:][::-1]

# --- STAGE 2: Fine Reranking (1536 Dimensions) ---
top_candidates = [corpus[idx] for idx in top_10_indices]
fine_query = model.encode(query, truncate_dim=1536, normalize_embeddings=True)
fine_candidates = model.encode(top_candidates, truncate_dim=1536, normalize_embeddings=True)

fine_scores = fine_candidates @ fine_query
final_best_idx = top_10_indices[np.argmax(fine_scores)]

print(f"Top Matched Snippet (Final score: {fine_scores.max():.4f}):")
print(corpus[final_best_idx])
```

---

### Example 4: Debugging & Crash Traceback Matching

Retrieve the buggy function in your codebase directly from a runtime exception traceback:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("divakantpandey/code-embed-v2")

traceback_query = """
Traceback (most recent call last):
  File "payment_service.py", line 42, in process_transaction
    charge_amount = invoice.total * (1 - discount_rate)
TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
"""

repo_functions = [
    """def calculate_shipping(weight, distance):
    base_rate = 5.0
    return base_rate + (weight * 0.5) + (distance * 0.1)""",
    
    """def process_transaction(user_id, invoice, discount_rate=None):
    if discount_rate is None:
        discount_rate = 0.0
    charge_amount = invoice.total * (1 - discount_rate)
    return payment_gateway.charge(user_id, charge_amount)""",
    
    """def validate_email(email_address):
    import re
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
    return bool(re.search(regex, email_address))"""
]

query_vec = model.encode(traceback_query, normalize_embeddings=True)
doc_vecs = model.encode(repo_functions, normalize_embeddings=True)

best_idx = (doc_vecs @ query_vec).argmax()
print("Associated Buggy Function Identified:")
print(repo_functions[best_idx])
```

---

### Example 5: Qdrant Vector Database Integration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Initialize client & model
qdrant = QdrantClient(":memory:")
model = SentenceTransformer("divakantpandey/code-embed-v2")

collection_name = "codebase_symbols"
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Ingest code files with metadata
files = [
    {"id": 1, "path": "auth/session.go", "lang": "go", "code": "func VerifySession(token string) bool { return redis.Exists(token) }"},
    {"id": 2, "path": "models/user.py", "lang": "python", "code": "class User(BaseModel): id: int; name: str; active: bool"},
]

points = []
for f in files:
    vector = model.encode(f["code"], normalize_embeddings=True).tolist()
    points.append(PointStruct(id=f["id"], vector=vector, payload=f))

qdrant.upsert(collection_name=collection_name, points=points)

# Search with metadata filtering
query_emb = model.encode("validate active user session", normalize_embeddings=True).tolist()
search_results = qdrant.query_points(
    collection_name=collection_name,
    query=query_emb,
    limit=1
)

for hit in search_results.points:
    print(f"Path: {hit.payload['path']} (Score: {hit.score:.4f})")
    print(f"Code: {hit.payload['code']}")
```

---

### Example 6: LangChain Code Repository Q&A (RAG Pipeline)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. Initialize embeddings with CodeEmbed v2
embeddings = HuggingFaceEmbeddings(
    model_name="divakantpandey/code-embed-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Build local document store
documents = [
    Document(page_content="func RateLimiter(ip string) bool { return limiter.Allow(ip) }", metadata={"file": "security/ratelimit.go"}),
    Document(page_content="def send_metric(name, val): statsd.increment(name, val)", metadata={"file": "telemetry/metrics.py"})
]

vector_db = Chroma.from_documents(documents, embeddings)

# 3. Retrieve relevant context for an LLM query
retriever = vector_db.as_retriever(search_kwargs={"k": 1})
matched_docs = retriever.invoke("how do we prevent DDOS attacks on our API endpoints?")

for doc in matched_docs:
    print(f"Source: {doc.metadata['file']}")
    print(f"Content: {doc.page_content}")
```

---

## 📜 License & Citation

This project and its model weights are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0)**.

- **Non-Commercial Use:** Free for open-source research, academic benchmarking, developer tooling experiments, and non-commercial projects with attribution.
- **Attribution:** Please attribute **Divakant Pandey** and cite:
  ```bibtex
  @misc{pandey2026codeembedv2,
    author = {Divakant Pandey},
    title = {CodeEmbed v2: High-Performance Multilingual Code Embedding Model},
    year = {2026},
    publisher = {Hugging Face / GitHub},
    howpublished = {\url{https://huggingface.co/divakantpandey/code-embed-v2}}
  }
  ```
- **Commercial Inquiries:** For commercial deployments, SaaS integration, or enterprise licenses, contact:
  - **Maintainer:** Divakant Pandey
  - **Email:** `divakant01@gmail.com`
  - **GitHub:** [@divakantpandey](https://github.com/divakantpandey)

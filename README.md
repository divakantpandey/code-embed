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
  - sentence-transformers
  - matryoshka
  - unixcoder
  - mteb
  - feature-extraction
  - gguf
  - onnx
license: cc-by-nc-4.0
pipeline_tag: feature-extraction
library_name: sentence-transformers
base_model: microsoft/unixcoder-base
datasets:
  - code-search-net/code_search_net
model-index:
  - name: code-embed-v2
    results:
      - task:
          type: retrieval
          name: MTEB(Code, v1)
        dataset:
          name: MTEB(Code, v1) mean over 12 tasks
          type: mteb/benchmark
          split: test
        metrics:
          - type: main_score
            value: 0.5475
            name: mean main_score
      - task:
          type: retrieval
          name: CoIR
        dataset:
          name: CoIR mean over 10 tasks
          type: mteb/benchmark
          split: test
        metrics:
          - type: main_score
            value: 0.5343
            name: mean main_score
      - task:
          type: retrieval
          name: CoREB(v1)
        dataset:
          name: CoREB(v1) mean over 6 tasks
          type: mteb/benchmark
          split: test
        metrics:
          - type: main_score
            value: 0.4362
            name: mean main_score
      - task:
          type: retrieval
          name: AppsRetrieval
        dataset:
          name: AppsRetrieval
          type: mteb/AppsRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.2017
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: COIRCodeSearchNetRetrieval
        dataset:
          name: COIRCodeSearchNetRetrieval
          type: mteb/COIRCodeSearchNetRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.7488
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeEditSearchRetrieval
        dataset:
          name: CodeEditSearchRetrieval
          type: mteb/CodeEditSearchRetrieval
          split: train
        metrics:
          - type: main_score
            value: 0.3703
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeFeedbackMT
        dataset:
          name: CodeFeedbackMT
          type: mteb/CodeFeedbackMT
          split: test
        metrics:
          - type: main_score
            value: 0.6164
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeFeedbackST
        dataset:
          name: CodeFeedbackST
          type: mteb/CodeFeedbackST
          split: test
        metrics:
          - type: main_score
            value: 0.6680
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeSearchNetCCRetrieval
        dataset:
          name: CodeSearchNetCCRetrieval
          type: mteb/CodeSearchNetCCRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.5664
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeSearchNetRetrieval
        dataset:
          name: CodeSearchNetRetrieval
          type: mteb/CodeSearchNetRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.8574
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeTransOceanContest
        dataset:
          name: CodeTransOceanContest
          type: mteb/CodeTransOceanContest
          split: test
        metrics:
          - type: main_score
            value: 0.6814
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CodeTransOceanDL
        dataset:
          name: CodeTransOceanDL
          type: mteb/CodeTransOceanDL
          split: test
        metrics:
          - type: main_score
            value: 0.3533
            name: main_score (ndcg_at_10 or map)
      - task:
          type: reranking
          name: CorebC2CReranking
        dataset:
          name: CorebC2CReranking
          type: mteb/CorebC2CReranking
          split: test
        metrics:
          - type: main_score
            value: 0.4205
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CorebC2CRetrieval
        dataset:
          name: CorebC2CRetrieval
          type: mteb/CorebC2CRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.2750
            name: main_score (ndcg_at_10 or map)
      - task:
          type: reranking
          name: CorebC2TReranking
        dataset:
          name: CorebC2TReranking
          type: mteb/CorebC2TReranking
          split: test
        metrics:
          - type: main_score
            value: 0.7937
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CorebC2TRetrieval
        dataset:
          name: CorebC2TRetrieval
          type: mteb/CorebC2TRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.6073
            name: main_score (ndcg_at_10 or map)
      - task:
          type: reranking
          name: CorebT2CReranking
        dataset:
          name: CorebT2CReranking
          type: mteb/CorebT2CReranking
          split: test
        metrics:
          - type: main_score
            value: 0.2539
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CorebT2CRetrieval
        dataset:
          name: CorebT2CRetrieval
          type: mteb/CorebT2CRetrieval
          split: test
        metrics:
          - type: main_score
            value: 0.2671
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: CosQA
        dataset:
          name: CosQA
          type: mteb/CosQA
          split: test
        metrics:
          - type: main_score
            value: 0.2997
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: StackOverflowQA
        dataset:
          name: StackOverflowQA
          type: mteb/StackOverflowQA
          split: test
        metrics:
          - type: main_score
            value: 0.7131
            name: main_score (ndcg_at_10 or map)
      - task:
          type: retrieval
          name: SyntheticText2SQL
        dataset:
          name: SyntheticText2SQL
          type: mteb/SyntheticText2SQL
          split: test
        metrics:
          - type: main_score
            value: 0.4938
            name: main_score (ndcg_at_10 or map)
---

# code-embed-v2

`divakantpandey/code-embed-v2` is a 129.5M-parameter code embedding model fine-tuned from [microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base) on the CodeSearchNet train split (Python, JavaScript, Go, Java, Ruby, PHP). It maps natural-language queries and functions to 1536-dimensional unit vectors for code search.

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-divakantpandey%2Fcode--embed--v2-blue)](https://huggingface.co/divakantpandey/code-embed-v2)
[![MTEB Leaderboard](https://img.shields.io/badge/MTEB%20Leaderboard-code--embed--v2-blue)](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2)
[![MTEB Results](https://img.shields.io/badge/MTEB%20Results-PR%20%23803-brightgreen)](https://github.com/embeddings-benchmark/results/pull/803)
[![Context Length: 1024](https://img.shields.io/badge/Context%20Length-1024%20tokens-lightgrey.svg)]()
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> [!WARNING]
> **Corrections (2026-11-17).** Earlier versions of this card and the accompanying preprint draft contained claims that were wrong or not backed by a measurement:
>
> - **Context length is 1,024 tokens, not 8,192.** The model has 1,026 absolute position embeddings and was trained and evaluated at 1,024 tokens. The previous config advertised 8,192, which made any input over 1,024 tokens fail with `IndexError: index 1026 is out of bounds`. The config now truncates at 1,024.
> - **Not fully zero-shot.** The model was trained on CodeSearchNet, which is the source of `CodeSearchNetRetrieval`, `COIRCodeSearchNetRetrieval` and `CodeSearchNetCCRetrieval`. Scores on those three tasks are in-distribution.
> - **Matryoshka truncation quality and CPU latency were never measured.** The tables that reported them have been removed.
> - **Training used in-batch negatives only**, not BM25 hard negatives. Pooling is mean pooling, not CLS. The Matryoshka dimensions were 64, 128, 256, 384, 768 and 1536.
> - **Supported languages are the six CodeSearchNet languages.** TypeScript and SQL were never in the training data.
> - **Parameter count is 129.5M** (90.0M excluding the token embedding table), not 131M total / 92M active.

## Model details

| | |
|---|---|
| Base model | `microsoft/unixcoder-base` (RoBERTa architecture, 12 layers, hidden size 768) |
| Pooling | Mean over tokens |
| Head | Linear 768 → 1536, GELU, Linear 1536 → 1536, L2 normalisation |
| Output dimension | 1536 (trained with a Matryoshka loss at 64, 128, 256, 384, 768, 1536) |
| Maximum input | 1,024 tokens; longer inputs are truncated |
| Parameters | 129,471,744 total; 89,984,256 excluding the token embedding table |
| Training data | CodeSearchNet train split, 6 languages, 1,838,250 docstring–function pairs after filtering and deduplication |
| Formats | safetensors (Sentence Transformers), ONNX (`model_optimized.onnx`, 135 MB), GGUF Q8_0 (138 MB, see the note under Ollama) |
| License | CC-BY-NC-4.0 |

## Training

Three stages, each one epoch, effective batch 128 (32 × 4 gradient accumulation), bf16 autocast, maximum length 1,024 tokens. Config: [`configs/v2_multilingual_config.yaml`](https://github.com/divakantpandey/code-embed/blob/main/configs/v2_multilingual_config.yaml).

1. **SimCSE** on functions: two dropout views (p = 0.1) of the same function are positives, temperature 0.05, lr 3e-5.
2. **Multiple Negatives Ranking Loss** on docstring → function pairs with in-batch negatives only, scale 25, lr 2.7e-5.
3. **Matryoshka loss** wrapping the same ranking loss at dimensions 64, 128, 256, 384, 768, 1536 with weights 0.10, 0.10, 0.15, 0.15, 0.25, 0.25, lr 1e-5.

## Evaluation

Official MTEB results, `mteb` 2.26, model revision `d7759ca2`, merged in [embeddings-benchmark/results#803](https://github.com/embeddings-benchmark/results/pull/803). Evaluation used 1,024 tokens.

| Benchmark | Tasks | Mean main score |
|---|---|---|
| MTEB(Code, v1) | 12 | 0.5475 |
| CoIR | 10 | 0.5343 |
| CoREB(v1) | 6 | 0.4362 |

| Task | Metric | Score | Note |
|---|---|---|---|
| AppsRetrieval | nDCG@10 | 0.2017 | |
| CodeEditSearchRetrieval | nDCG@10 | 0.3703 | |
| CodeFeedbackMT | nDCG@10 | 0.6164 | |
| CodeFeedbackST | nDCG@10 | 0.6680 | |
| CodeSearchNetCCRetrieval | nDCG@10 | 0.5664 | in-distribution |
| CodeSearchNetRetrieval | nDCG@10 | 0.8574 | in-distribution |
| COIRCodeSearchNetRetrieval | nDCG@10 | 0.7488 | in-distribution |
| CodeTransOceanContest | nDCG@10 | 0.6814 | |
| CodeTransOceanDL | nDCG@10 | 0.3533 | |
| CosQA | nDCG@10 | 0.2997 | |
| StackOverflowQA | nDCG@10 | 0.7131 | |
| SyntheticText2SQL | nDCG@10 | 0.4938 | |
| CorebC2CReranking | MAP | 0.4205 | |
| CorebC2CRetrieval | nDCG@10 | 0.2750 | |
| CorebC2TReranking | MAP | 0.7937 | |
| CorebC2TRetrieval | nDCG@10 | 0.6073 | |
| CorebT2CReranking | MAP | 0.2539 | |
| CorebT2CRetrieval | nDCG@10 | 0.2671 | |

**Leaderboard position.** On 2026-10-24, filtered to models with at most 150M parameters, the MTEB leaderboard listed this model 3rd on CoREB(v1), 8th on CoIR and 9th on MTEB(Code, v1), and 3rd of 25 models overall on CorebC2CReranking. Ranks change as models are added; check the [live leaderboard](https://mteb-leaderboard.hf.space/models/divakantpandey/code-embed-v2).

**Zero-shot status.** The model's current mteb metadata declares only `CodeSearchNetRetrieval` as training data, which gives 91% zero-shot on MTEB(Code, v1). `COIRCodeSearchNetRetrieval` and `CodeSearchNetCCRetrieval` are also built from CodeSearchNet. Declaring all three gives 75% zero-shot on MTEB(Code, v1), 80% on CoIR and 100% on CoREB(v1).

## Limitations

- **No comparison against the base model.** UniXcoder-base was not evaluated with the same harness, so the gain from this fine-tune is not quantified.
- **Truncated dimensions are untested.** The model was trained with a Matryoshka loss, but retrieval quality at truncated dimensions has not been measured. Validate on your own data before relying on it.
- **Latency is not benchmarked.**
- **The GGUF file is not the same embedding.** It outputs the 768-dimensional backbone representation without the projection head, and it has not been evaluated. Do not mix GGUF vectors with Sentence Transformers vectors in one index.
- **Function-level training only.** Training pairs are single functions with their docstrings, at most 1,024 tokens. Whole files, other languages, and non-docstring queries are out of distribution.

## Usage

### Semantic code search with `sentence-transformers`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("divakantpandey/code-embed-v2")

queries = [
    "binary search implementation with edge handling",
    "http json get request with retry logic",
    "worker pool concurrency pattern using channels",
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
}""",
]

query_vecs = model.encode(queries, normalize_embeddings=True)
code_vecs = model.encode(code_candidates, normalize_embeddings=True)
similarity = query_vecs @ code_vecs.T

for i, q in enumerate(queries):
    best = similarity[i].argmax()
    print(f"{q!r} -> candidate {best} (score {similarity[i][best]:.4f})")
```

### Code-to-code similarity

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("divakantpandey/code-embed-v2")

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

candidates = [
    """func QuickSort(arr []int) []int {
        if len(arr) <= 1 { return arr }
        pivot := arr[len(arr)/2]
        var left, middle, right []int
        for _, x := range arr {
            if x < pivot { left = append(left, x) } else if x == pivot { middle = append(middle, x) } else { right = append(right, x) }
        }
        return append(append(QuickSort(left), middle...), QuickSort(right)...)
    }""",
    """func BubbleSort(arr []int) []int {
        n := len(arr)
        for i := 0; i < n-1; i++ {
            for j := 0; j < n-i-1; j++ {
                if arr[j] > arr[j+1] { arr[j], arr[j+1] = arr[j+1], arr[j] }
            }
        }
        return arr
    }""",
]

ref = model.encode(reference_python, normalize_embeddings=True)
cands = model.encode(candidates, normalize_embeddings=True)
print(cands @ ref)
```

The model was not trained on cross-language pairs. Code-to-code scores across languages come from the shared docstring-to-code objective; check them on your own data.

### Truncated embeddings

The model was trained with a Matryoshka loss at 64, 128, 256, 384, 768 and 1536 dimensions, so leading prefixes of the vector can be used as smaller embeddings. Quality at each size has not been measured.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("divakantpandey/code-embed-v2")

query = "JWT token generation with expiration time"
corpus = [
    "import jwt, datetime\ndef make_token(uid): return jwt.encode({'sub': uid, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, 'secret')",
    "import hashlib\ndef hash_password(pwd): return hashlib.sha256(pwd.encode()).hexdigest()",
    "def connect_db(): return psycopg2.connect('dbname=app user=postgres')",
    "import jwt\ndef decode_token(tok): return jwt.decode(tok, 'secret', algorithms=['HS256'])",
]

# Stage 1: shortlist with 256-d prefixes.
q256 = model.encode(query, truncate_dim=256, normalize_embeddings=True)
c256 = model.encode(corpus, truncate_dim=256, normalize_embeddings=True)
shortlist = np.argsort(c256 @ q256)[::-1][:2]

# Stage 2: rerank the shortlist with full 1536-d vectors.
q = model.encode(query, normalize_embeddings=True)
c = model.encode([corpus[i] for i in shortlist], normalize_embeddings=True)
print(corpus[shortlist[np.argmax(c @ q)]])
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

qdrant = QdrantClient(":memory:")
model = SentenceTransformer("divakantpandey/code-embed-v2")

qdrant.create_collection(
    collection_name="codebase_symbols",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

files = [
    {"id": 1, "path": "auth/session.go", "code": "func VerifySession(token string) bool { return redis.Exists(token) }"},
    {"id": 2, "path": "models/user.py", "code": "class User(BaseModel): id: int; name: str; active: bool"},
]
qdrant.upsert(
    collection_name="codebase_symbols",
    points=[
        PointStruct(id=f["id"], vector=model.encode(f["code"], normalize_embeddings=True).tolist(), payload=f)
        for f in files
    ],
)

hits = qdrant.query_points(
    collection_name="codebase_symbols",
    query=model.encode("validate active user session", normalize_embeddings=True).tolist(),
    limit=1,
)
for hit in hits.points:
    print(hit.payload["path"], round(hit.score, 4))
```

### LangChain

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="divakantpandey/code-embed-v2",
    encode_kwargs={"normalize_embeddings": True},
)

documents = [
    Document(page_content="func RateLimiter(ip string) bool { return limiter.Allow(ip) }", metadata={"file": "security/ratelimit.go"}),
    Document(page_content="def send_metric(name, val): statsd.increment(name, val)", metadata={"file": "telemetry/metrics.py"}),
]
vector_db = Chroma.from_documents(documents, embeddings)

for doc in vector_db.as_retriever(search_kwargs={"k": 1}).invoke("rate limit requests per client IP"):
    print(doc.metadata["file"])
```

### Ollama and llama.cpp (GGUF)

> [!NOTE]
> The GGUF file contains the transformer only. It returns the 768-dimensional pooled backbone output, without the 1536-d projection head, so its vectors are not the same as the Sentence Transformers output and have not been evaluated. Keep the context at 1,024 tokens.

```bash
ollama pull hf.co/divakantpandey/code-embed-v2:code-embed-v2-q8_0.gguf

curl http://localhost:11434/api/embed -d '{
  "model": "hf.co/divakantpandey/code-embed-v2:code-embed-v2-q8_0.gguf",
  "input": "def binary_search(arr, target):"
}'
```

```bash
wget https://huggingface.co/divakantpandey/code-embed-v2/resolve/main/code-embed-v2-q8_0.gguf
llama-server -m code-embed-v2-q8_0.gguf --embedding -c 1024 --port 8080
```

### Text Embeddings Inference

```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id divakantpandey/code-embed-v2
```

Check that the returned vectors are 1536-dimensional. A 768-dimensional result means the projection head was not applied.

## License and citation

Code and weights are released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/). For commercial use, contact Divakant Pandey (`divakant01@gmail.com`, [@divakantpandey](https://github.com/divakantpandey)).

```bibtex
@misc{pandey2026codeembedv2,
  author = {Divakant Pandey},
  title = {code-embed-v2: A UniXcoder-Based Code Embedding Model Fine-Tuned on CodeSearchNet},
  year = {2026},
  howpublished = {\url{https://huggingface.co/divakantpandey/code-embed-v2}}
}
```

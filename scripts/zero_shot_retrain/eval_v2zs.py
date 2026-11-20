"""Score the zero-shot retrain with code-embed-v2's official harness settings (same pins as the UniXcoder
baseline): mteb 2.26.17, sentence-transformers 6.2.0, fp32, 1,024 tokens, encode batch 64."""

import sys
import time
from pathlib import Path

import mteb
from huggingface_hub import HfApi
from mteb.cache import ResultCache
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from sentence_transformers import SentenceTransformer

MODEL, OUT = sys.argv[1], Path("/workspace/results")
OUT.mkdir(parents=True, exist_ok=True)
model = SentenceTransformer(MODEL, device="cuda")
model.max_seq_length = 1024
wrapper = SentenceTransformerEncoderWrapper(model)
wrapper.mteb_model_meta = wrapper.mteb_model_meta.model_copy(
    update={"name": "divakantpandey/code-embed-v2-zeroshot"}
)
tasks, seen = [], set()
for b in ("MTEB(Code, v1)", "CoIR", "CoREB(v1)"):
    for t in mteb.get_benchmark(b).tasks:
        if t.metadata.name not in seen:
            seen.add(t.metadata.name)
            tasks.append(t)
for t in tasks:
    t0 = time.time()
    mteb.evaluate(
        wrapper, [t], encode_kwargs={"batch_size": 64}, cache=ResultCache(str(OUT / "cache"))
    )
    print(f"eval done {t.metadata.name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    for attempt in range(
        6
    ):  # host networking can drop for minutes; don't lose an eval to one failed call
        try:
            HfApi().upload_folder(
                folder_path=str(OUT),
                repo_id="divakantpandey/code-embed-v3-src",
                path_in_repo="v2-zeroshot/results",
                commit_message=f"v2-zeroshot eval {t.metadata.name}",
            )
            break
        except Exception as e:
            print(f"upload attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 5:
                raise
            time.sleep(120)
(OUT / "DONE").write_text("ok")

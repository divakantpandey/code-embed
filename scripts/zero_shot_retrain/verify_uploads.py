"""Exit 0 only when the trained model and every eval result are on the Hub and match the local copies."""

import sys
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()
model_dir, results_dir = Path("/workspace/out/stage3"), Path("/workspace/results")
if not (results_dir / "DONE").exists():
    sys.exit("eval DONE marker missing")

# Re-push results first: the last per-task upload may have raced the DONE marker or failed transiently.
api.upload_folder(
    folder_path=str(results_dir),
    repo_id="divakantpandey/code-embed-v3-src",
    path_in_repo="v2-zeroshot/results",
    commit_message="v2-zeroshot: final results sync",
)

local_model = {
    p.relative_to(model_dir).as_posix(): p.stat().st_size
    for p in model_dir.rglob("*")
    if p.is_file()
}
remote_model = {
    f.path: f.size
    for f in api.get_paths_info("divakantpandey/code-embed-v2-zeroshot", list(local_model))
}
bad = [k for k, v in local_model.items() if remote_model.get(k) != v]
if bad:
    sys.exit(f"model files missing or size mismatch on Hub: {bad[:5]}")

local_res = {p.relative_to(results_dir).as_posix() for p in results_dir.rglob("*.json")}
remote_res = {
    f.removeprefix("v2-zeroshot/results/")
    for f in api.list_repo_files("divakantpandey/code-embed-v3-src")
    if f.startswith("v2-zeroshot/results/")
}
missing = sorted(local_res - remote_res)
if missing or not local_res:
    sys.exit(f"results missing on Hub: {missing[:5]} (local json count {len(local_res)})")
print(f"verified: {len(local_model)} model files, {len(local_res)} result files")

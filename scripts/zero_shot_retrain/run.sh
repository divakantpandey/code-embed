#!/bin/bash
# Driver: build + audit (CPU) alongside a GPU smoke test, then train 3 stages, then the pinned MTEB eval.
# Every phase writes a marker, so a rerun resumes after the last finished phase.
set -uo pipefail
W=/workspace; M=$W/markers; mkdir -p $M $W/out
export PATH="$HOME/.local/bin:$PATH" HF_HOME=$W/.cache/huggingface UV_LINK_MODE=copy WANDB_PROJECT=code-embed
cd $W/code
note() {
  echo "{\"note\": \"$1\", \"time\": \"$(date -u +%H:%M:%S)\"}" > $W/out/progress.json
  cp -f $W/*.log $W/out/ 2>/dev/null
  uv run --no-sync python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='$W/out', repo_id='divakantpandey/code-embed-v3-src', path_in_repo='v2-zeroshot/run',
  allow_patterns=['*.json','*.log'], commit_message='v2-zeroshot: $1')" >/dev/null 2>&1 || echo "upload failed: $1"
}
fail() { note "FAILED $1"; exit 1; }
# Host networking on Vast can drop for minutes; a single failed Hub call must not end the run.
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "attempt $i failed: $1" >&2; sleep 120; done; return 1; }
note "booted"
# The guard must start after "booted" overwrites progress.json, or it would act on a stale FAILED note.
pgrep -f "bash $W/guard.sh" >/dev/null || { setsid nohup bash $W/guard.sh >/dev/null 2>&1 < /dev/null & }
# Check the packages, not the folder: an interrupted install leaves a venv without them.
evenv_ok() { $W/evenv/bin/python -c "import mteb, sentence_transformers, torch" 2>/dev/null; }
install_evenv() { rm -rf $W/evenv && uv venv -q -p 3.12 $W/evenv && uv pip install -q -p $W/evenv "torch==2.6.0" "mteb==2.26.17" "sentence-transformers==6.2.0" "transformers==5.20.2"; }
evenv_ok || ( retry install_evenv > $W/evenv.log 2>&1 ) &
EVENV=$!
if [ ! -f $M/data ]; then
  ( uv run --no-sync python $W/train_v2zs.py --smoke --out $W/out > $W/smoke.log 2>&1 && touch $M/smoke ) &
  SMOKE=$!
  uv run --no-sync ce3 build-data $W/v2_zeroshot.yaml > $W/build.log 2>&1 || fail build
  note "built"
  wait $SMOKE; [ -f $M/smoke ] || fail smoke
  grep "SMOKE OK" $W/smoke.log > $W/out/smoke.txt
  note "smoke ok"
  uv run --no-sync ce3 audit $W/data/v2_zeroshot --no-external > $W/audit.log 2>&1 || fail audit
  cp $W/data/v2_zeroshot/contamination_report.json $W/data/v2_zeroshot/manifest.json $W/out/ 2>/dev/null
  touch $M/data; note "audited"
fi
if [ ! -f $M/train ]; then
  uv run --no-sync python $W/train_v2zs.py --out $W/out >> $W/train.log 2>&1 || fail train
  touch $M/train; note "trained"
fi
if [ ! -f $M/upload ]; then
  retry uv run --no-sync python -c "
from huggingface_hub import HfApi
api = HfApi(); r = 'divakantpandey/code-embed-v2-zeroshot'
api.create_repo(r, private=True, exist_ok=True)
api.upload_folder(folder_path='$W/out/stage3', repo_id=r, commit_message='Zero-shot retrain of the code-embed-v2 recipe')" || fail upload
  touch $M/upload; note "model uploaded"
fi
wait $EVENV
evenv_ok || fail evenv
if [ ! -f $M/eval ]; then
  note "eval started"
  $W/evenv/bin/python $W/eval_v2zs.py $W/out/stage3 >> $W/eval.log 2>&1 || fail eval
  touch $M/eval
fi
note "all done"

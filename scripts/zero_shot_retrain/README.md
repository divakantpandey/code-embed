# Zero-shot retrain of the code-embed-v2 recipe

v2's recipe on CoRNStack instead of CodeSearchNet, at v2's per-language row counts, audited against all 18
eval tasks. Results and discussion: `docs/zero_shot_retrain.md`.

The data build, the contamination audit and `load_mix` come from
[code-embed-v3](https://github.com/divakantpandey/code-embed-v3) at `d34068d`, which `onstart.sh` clones on the
instance; nothing else from v3 is used.

| File | Role |
|---|---|
| `v2_zeroshot.yaml` | Data config for `ce3 build-data` (6 CoRNStack languages) |
| `build_fast.py` | Parallel shard download and build, used instead of streaming `build-data` |
| `train_v2zs.py` | The three training stages (SimCSE, MNRL, MNRL + Matryoshka); `--smoke` for a short check |
| `eval_v2zs.py` | 18-task eval with v2's harness pins; uploads each task's results to the Hub |
| `onstart.sh`, `run.sh` | Instance boot script and resumable driver (data, train, upload, eval) |
| `guard.sh`, `verify_uploads.py` | On-instance guard: destroys the instance once the uploads are verified, stops it on failure |
| `watch.sh` | Local status log, with a fallback destroy |
| `three_way.py` | Builds the UniXcoder-base / v2 / retrain table from `unixcoder_scores.json` and the eval results |
| `contamination_report.json`, `three_way.json` | Audit output and the per-task scores |
| `results/unixcoder-base/`, `results/code-embed-v2-zeroshot/` | Raw MTEB result files for the base model and the retrain, copied from [code-embed-v3-src@b3a8737](https://huggingface.co/divakantpandey/code-embed-v3-src/tree/b3a8737180d583f4595428e82efc7cd0bfcf4eb8) (`baselines/unixcoder-base/`, `v2-zeroshot/results/`) |

`python three_way.py unixcoder_scores.json results/code-embed-v2-zeroshot three_way.json`

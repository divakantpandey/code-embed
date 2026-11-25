# code-embed-v3 at d34068d (vendored subset)

The zero-shot retrain in `scripts/zero_shot_retrain/` built its CoRNStack data and ran the contamination
audit with tools from [code-embed-v3](https://github.com/divakantpandey/code-embed-v3) at commit
`d34068d3a850c8cf32d88cce5e1c83aca7d6d8d9`: `ce3 build-data`, `ce3 audit` and `code_embed_v3.data.build.load_mix`.
This folder holds the modules those three entry points import, copied byte for byte from that commit, so the
retrain can be read and rerun from this repository alone. Apache-2.0, see `LICENSE`.

| Module | Used for |
|---|---|
| `data/sources.py`, `data/build.py`, `config.py`, `prompts.py` | `ce3 build-data` and `load_mix` |
| `data/contamination.py` | `ce3 audit`: exact and MinHash near-duplicate matching against the eval task corpora |
| `cli.py`, `env.py`, `paths.py`, `modeling.py`, `eval/run_mteb.py`, `eval/behaviour_eval.py` | The `ce3` CLI and the imports `audit` pulls in (benchmark task lists) |

`cli.py` also defines commands (`mine`, `label`, `train`, `merge`, ...) whose modules are not included here;
only `build-data` and `audit` work from this copy. `pyproject.toml` and `uv.lock` are v3's own, unchanged.

Requires Python 3.12+:

```bash
cd vendor/code-embed-v3
uv sync --frozen
uv run ce3 build-data ../../scripts/zero_shot_retrain/v2_zeroshot.yaml
uv run ce3 audit /workspace/data/v2_zeroshot --no-external
```

`build-data` writes to the `output_dir` in the YAML (`/workspace/data/v2_zeroshot`, as on the training
instance); edit it to build elsewhere. These are the two commands `scripts/zero_shot_retrain/run.sh` runs.

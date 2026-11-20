"""Parallel CoRNStack build for the v2-zeroshot mix.

Streaming the Hub shards ran at ~150 rows/s (3+ h for 1.84M rows) with the GPU idle. Each shard holds
100k rows, so this downloads only the shards each language needs, builds the six sources in parallel
with v3's own build_mix (same filters, dedup and dev hold-out), and merges them into one mix dir."""

from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

W = Path("/workspace")
SHARDS, PARTS, OUT = W / "shards", W / "data" / "parts", W / "data" / "v2_zeroshot"
ROWS_PER_SHARD = 100_000
cfg = yaml.safe_load((W / "v2_zeroshot.yaml").read_text())


def lang(src: dict) -> str:
    return src["hf_id"].split("cornstack-")[1].removesuffix("-v1")


def files(src: dict) -> list[str]:
    # One spare shard's worth of headroom is not needed: filters and dedup drop well under 1%.
    n = math.ceil(src["max_rows"] / ROWS_PER_SHARD)
    return [f"shard-{i:05d}.jsonl.gz" for i in range(n)]


def fetch(args: tuple[str, str]) -> str:
    repo, f = args
    return hf_hub_download(
        repo, f, repo_type="dataset", local_dir=str(SHARDS / repo.split("cornstack-")[1])
    )


def build(src: dict) -> str:
    from code_embed_v3.config import DataMixConfig
    from code_embed_v3.data.build import build_mix

    local = sorted(str(p) for p in (SHARDS / f"{lang(src)}-v1").glob("shard-*.jsonl.gz"))
    one = {
        **cfg,
        "output_dir": str(PARTS / src["name"]),
        "sources": [{**{k: v for k, v in src.items() if k != "hf_id"}, "data_files": local}],
    }
    p = PARTS / f"{src['name']}.yaml"
    p.write_text(yaml.safe_dump(one))
    build_mix(DataMixConfig.from_yaml(p))
    for f in local:
        Path(f).unlink()
    return src["name"]


if __name__ == "__main__":
    PARTS.mkdir(parents=True, exist_ok=True)
    jobs = [(s["hf_id"], f) for s in cfg["sources"] for f in files(s)]
    print(f"downloading {len(jobs)} shards", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for path in ex.map(fetch, jobs):
            print("got", path, flush=True)
    with ProcessPoolExecutor(len(cfg["sources"])) as ex:
        for name in ex.map(build, cfg["sources"]):
            print("built", name, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev").mkdir(exist_ok=True)
    manifest: dict = {"sources": {}}
    for s in cfg["sources"]:
        part = PARTS / s["name"]
        shutil.move(str(part / s["name"]), str(OUT / s["name"]))
        shutil.move(str(part / "dev" / s["name"]), str(OUT / "dev" / s["name"]))
        manifest["sources"].update(json.loads((part / "manifest.json").read_text())["sources"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({n: m["rows"] for n, m in manifest["sources"].items()}, flush=True)

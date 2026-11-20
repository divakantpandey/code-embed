"""code-embed-v2's three-stage recipe on zero-shot data, re-implemented in Sentence Transformers.

Architecture and hyperparameters copy the released v2 checkpoint and configs/v2_multilingual_config.yaml:
UniXcoder-base, mean pooling, Dense 768->1536 GELU, Dense 1536->1536, Normalize; 1,024 tokens; bf16;
per-device batch 32 x 4 accumulation; one epoch per stage. Gradient accumulation does not add in-batch
negatives, so each step sees 31 negatives, as in v2.

--smoke runs every stage for a few steps on streamed rows and prints samples/s, so the full run's time
and cost can be projected before it starts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, Normalize, Pooling, Transformer
from sentence_transformers.sentence_transformer.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

BASE = "microsoft/unixcoder-base"
DIMS = [64, 128, 256, 384, 768, 1536]
DIM_WEIGHTS = [0.1, 0.1, 0.15, 0.15, 0.25, 0.25]
STAGES = {
    1: {"lr": 3.0e-5, "warmup": 0.10, "sched": "cosine", "scale": 1 / 0.05},
    2: {"lr": 2.6975e-5, "warmup": 0.15, "sched": "cosine_with_restarts", "scale": 25.0},
    3: {"lr": 1.0e-5, "warmup": 0.10, "sched": "cosine", "scale": 25.0},
}


def build_model() -> SentenceTransformer:
    tr = Transformer(BASE, max_seq_length=1024)
    pool = Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean")
    d1 = Dense(768, 1536, activation_function=torch.nn.GELU())
    d2 = Dense(1536, 1536, activation_function=torch.nn.Identity())
    return SentenceTransformer(modules=[tr, pool, d1, d2, Normalize()])


def load_pairs(data_dir: str) -> Dataset:
    from code_embed_v3.data.build import load_mix

    mix, _ = load_mix(data_dir, require_audit=True)
    parts = [mix[k].select_columns(["anchor", "positive"]) for k in sorted(mix)]
    return concatenate_datasets(parts).shuffle(seed=42)


def smoke_pairs(n: int) -> Dataset:
    from datasets import load_dataset

    rows = []
    for lang, k in (
        ("php", 3),
        ("java", 3),
        ("python", 2),
        ("go", 2),
        ("javascript", 1),
        ("ruby", 1),
    ):
        it = load_dataset(f"nomic-ai/cornstack-{lang}-v1", split="train", streaming=True)
        for i, r in enumerate(it):
            if i >= n * k // 12:
                break
            rows.append({"anchor": r["query"], "positive": r["document"]})
    return Dataset.from_list(rows).shuffle(seed=42)


class Throughput(TrainerCallback):
    def __init__(self, path: Path, stage: int, eff_batch: int) -> None:
        self.path, self.stage, self.eff = path, stage, eff_batch
        self.t0: float | None = None

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 5:
            self.t0 = time.time()
        if self.t0 and state.global_step % 50 == 0:
            sps = (state.global_step - 5) * self.eff / (time.time() - self.t0)
            left = (state.max_steps - state.global_step) * self.eff / sps / 3600
            self.path.write_text(
                json.dumps(
                    {
                        "stage": self.stage,
                        "step": state.global_step,
                        "max_steps": state.max_steps,
                        "samples_per_s": round(sps, 1),
                        "stage_hours_left": round(left, 2),
                    }
                )
            )


def run_stage(
    model: SentenceTransformer, stage: int, pairs: Dataset, out: Path, max_steps: int, report: str
) -> float:
    cfg = STAGES[stage]
    ds = pairs.map(lambda b: {"anchor": b["positive"]}, batched=True) if stage == 1 else pairs
    inner = MultipleNegativesRankingLoss(model, scale=cfg["scale"])
    loss = (
        MatryoshkaLoss(model, inner, matryoshka_dims=DIMS, matryoshka_weights=DIM_WEIGHTS)
        if stage == 3
        else inner
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / f"ckpt{stage}"),
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=cfg["lr"],
        warmup_steps=cfg["warmup"],
        lr_scheduler_type=cfg["sched"],
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        dataloader_num_workers=8,
        report_to=report,
        run_name=f"v2-zeroshot-stage{stage}",
        seed=42,
    )
    tp = Throughput(out / "throughput.json", stage, 128)
    t0 = time.time()
    SentenceTransformerTrainer(
        model=model, args=args, train_dataset=ds, loss=loss, callbacks=[tp]
    ).train()
    hours = (time.time() - t0) / 3600
    if max_steps < 0:
        model.save(str(out / f"stage{stage}"))
    return hours


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/data/v2_zeroshot")
    p.add_argument("--out", default="/workspace/out")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model()
    print(model, flush=True)
    if a.smoke:
        pairs = smoke_pairs(128 * 60)
        res = {}
        for s in (1, 2, 3):
            t0 = time.time()
            run_stage(model, s, pairs, out / "smoke", max_steps=40, report="none")
            res[s] = round((time.time() - t0), 1)
            print(f"smoke stage {s}: 40 steps in {res[s]} s", flush=True)
        print("SMOKE OK", json.dumps(res), flush=True)
        return
    pairs = load_pairs(a.data_dir)
    print(f"{len(pairs):,} training pairs", flush=True)
    done = json.loads((out / "stages.json").read_text()) if (out / "stages.json").exists() else {}
    for s in (1, 2, 3):
        if str(s) in done:
            model = SentenceTransformer(str(out / f"stage{s}"), device="cuda")
            continue
        h = run_stage(model, s, pairs, out, max_steps=-1, report="wandb")
        done[str(s)] = round(h, 3)
        (out / "stages.json").write_text(json.dumps(done))
        print(f"stage {s} done in {h:.2f} h", flush=True)


if __name__ == "__main__":
    main()

# Zero-shot retrain of the code-embed-v2 recipe (2026-11-19/20, Vast.ai 1x A100 SXM4 40 GB, $10.98)

**Why.** code-embed-v2 trained on CodeSearchNet, so three of its 18 MTEB tasks (CodeSearchNetRetrieval,
COIRCodeSearchNetRetrieval, CodeSearchNetCCRetrieval) are not zero-shot (mteb PR #5942 declares all three). Question:
how much of v2's gain over its base (UniXcoder-base) survives when no CodeSearchNet data is used?

**Procedure.** v2's recipe unchanged (UniXcoder-base, mean pooling, Dense 768->1536->1536, 1,024 tokens; stage 1
SimCSE, stage 2 MNRL scale 25, stage 3 MNRL + Matryoshka; batch 32 x accum 4, bf16). Data swapped: CoRNStack
(`nomic-ai/cornstack-{lang}-v1`) at v2's exact per-language row counts, 1,837,000 rows, then `ce3 audit` against all
18 tasks removed 4,272 rows (0.23%; ruby 3%), leaving 1,832,728. Hits by task: COIRCodeSearchNet 3,721,
CodeSearchNet 2,058, CodeSearchNetCC 78, Apps 13, CodeFeedbackST 3, CodeEdit 1. Stages took 6.8 h, 4.0 h, 4.0 h.
Data build and audit used the tools from code-embed-v3 at `d34068d` (`ce3 build-data`, `ce3 audit`).
Eval with v2's harness pins (mteb 2.26.17, sentence-transformers 6.2.0, fp32, 1,024 tokens, batch 64, no prompts).
Model: [`divakantpandey/code-embed-v2-zeroshot`](https://huggingface.co/divakantpandey/code-embed-v2-zeroshot); raw results: `scripts/zero_shot_retrain/results/` (copied from `code-embed-v3-src/v2-zeroshot/results`).
UniXcoder-base and the retrain are our runs with this harness; the v2 column is v2's official merged results.

| Benchmark | UniXcoder-base | v2 (trained on CSN) | v2 zero-shot retrain | retrain - v2 | retrain - base |
|---|---|---|---|---|---|
| MTEB(Code, v1), 12 tasks | 36.57 | 54.75 | 48.62 | -6.13 | +12.05 |
| CoIR, 10 tasks | 33.56 | 53.43 | 44.80 | -8.62 | +11.24 |
| CoREB(v1), 6 tasks | 35.31 | 43.62 | 28.99 | -14.64 | -6.32 |
| 3 CodeSearchNet-derived tasks | 56.83 | 72.42 | 71.82 | -0.60 | +15.00 |
| 15 other tasks | 32.02 | 46.77 | 36.13 | -10.64 | +4.11 |

Per task (retrain vs v2): above v2 on CodeEditSearch (+11.07), CosQA (+2.47), CodeSearchNetRetrieval (+1.55); largest
drops CorebC2TRetrieval (-28.64), CodeTransOceanContest (-24.42), CodeFeedbackMT (-23.00), CorebC2TReranking
(-19.45), CorebC2CRetrieval (-17.82), AppsRetrieval (-13.65), SyntheticText2SQL (-10.44). The retrain beats the base
on 11/18 tasks and is below it on all of CoREB's retrieval-from-code tasks (C2C -19.64, C2T -11.95).

**Findings (measured).**
- v2's scores on the CodeSearchNet-derived tasks are reproduced without any CodeSearchNet data (71.82 vs 72.42), so
  they are not an artefact of training on the benchmark's source.
- v2's broad gain on the other 15 tasks is not reproduced: +4.1 over base instead of +14.8. The recipe alone does not
  explain v2's results; the CodeSearchNet training data does a lot of the work off-benchmark too.
- On CoREB the retrain is worse than the untouched base (-6.3).

**Hypotheses (not tested).** CoRNStack's queries (docstrings mined at scale, then consistency-filtered) are noisier
or narrower than CodeSearchNet's curated function/docstring pairs; the hyperparameters were tuned for CodeSearchNet
and may not transfer; the 0.23% removed by the audit is too small to matter.

**Cost and ops.** $10.98 total (A100 at $0.527/h plus storage). Lost time: a `vastai start` on the running instance
restarted the container and threw away 1.9 h of stage 2 (training resumed only per stage, not per checkpoint); the
model upload failed once on host networking and the eval env was empty because the uv cache was deleted during its
install (both fixed with retries and an import check); ~40 min idle after completion because the on-instance guard
called `vastai destroy` without `-y`. All of these are now rules for remote jobs.

Scripts and per-task scores: `scripts/zero_shot_retrain/`.

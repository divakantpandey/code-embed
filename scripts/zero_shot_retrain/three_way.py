import json
import sys
from pathlib import Path

import mteb

v2 = {
    "AppsRetrieval": 20.17,
    "CosQA": 29.97,
    "SyntheticText2SQL": 49.38,
    "COIRCodeSearchNetRetrieval": 74.88,
    "CodeSearchNetCCRetrieval": 56.64,
    "CodeTransOceanContest": 68.14,
    "CodeTransOceanDL": 35.33,
    "StackOverflowQA": 71.31,
    "CodeFeedbackST": 66.80,
    "CodeFeedbackMT": 61.64,
    "CodeEditSearchRetrieval": 37.03,
    "CodeSearchNetRetrieval": 85.74,
    "CorebC2CReranking": 42.05,
    "CorebC2CRetrieval": 27.50,
    "CorebC2TReranking": 79.37,
    "CorebC2TRetrieval": 60.73,
    "CorebT2CReranking": 25.39,
    "CorebT2CRetrieval": 26.71,
}
ux = {k: 100 * v for k, v in json.loads(Path(sys.argv[1]).read_text()).items()}
zs = {}
for f in Path(sys.argv[2]).rglob("*.json"):
    r = json.loads(f.read_text())
    if "task_name" not in r:
        continue
    s = [e["main_score"] for split in r["scores"].values() for e in split]
    zs[r["task_name"]] = 100 * sum(s) / len(s)
assert set(zs) == set(v2), set(v2) ^ set(zs)
CSN = {"CodeSearchNetRetrieval", "COIRCodeSearchNetRetrieval", "CodeSearchNetCCRetrieval"}
rows = {t: (ux[t], v2[t], zs[t]) for t in v2}
Path(sys.argv[3]).write_text(json.dumps(rows, indent=1))
print(f"{'task':28s} {'unix':>6s} {'v2':>6s} {'zs':>6s} {'zs-v2':>6s} {'zs-ux':>6s}")
for t, (a, b, c) in rows.items():
    print(
        f"{t + (' *' if t in CSN else ''):28s} {a:6.2f} {b:6.2f} {c:6.2f} {c - b:+6.2f} {c - a:+6.2f}"
    )
groups = [
    (b, [t.metadata.name for t in mteb.get_benchmark(b).tasks])
    for b in ("MTEB(Code, v1)", "CoIR", "CoREB(v1)")
]
groups += [
    ("15 non-CSN tasks", [t for t in v2 if t not in CSN]),
    ("3 CSN-derived tasks", sorted(CSN)),
]
for name, ts in groups:
    m = [sum(rows[t][i] for t in ts) / len(ts) for i in range(3)]
    label = f"{name} (n={len(ts)})"
    print(
        f"{label:28s} {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[2] - m[1]:+6.2f} {m[2] - m[0]:+6.2f}"
    )
w = sum(rows[t][2] > rows[t][0] for t in v2)
print(f"zs beats unixcoder on {w}/18; beats v2 on {sum(rows[t][2] > rows[t][1] for t in v2)}/18")

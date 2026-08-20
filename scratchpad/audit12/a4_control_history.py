"""Audit 12 / A4: does the E42 control arm replicate?

Three independent measurements of the SAME recipe (20k episodes vs 3x rule_based,
warm-started from q_table_parent, evaluated at eps=0 vs 3x rule_based, val seed 550731):

  1. E37 PLB2 s100-114 @ ep20000, evaluated 2026-08-16  (15 seeds)
  2. E38 s120-124      @ ep20000, evaluated 2026-08-16/17 (5 seeds, 1000 rounds,
     trained AFTER the a76269b cleanup -- the code E42's treatment ran on)
  3. E42 ctl s100-107  @ ep20000, evaluated 2026-08-19 (8 seeds) -- the SAME TABLES
     as (1)'s first eight, re-evaluated three days later.

(1) vs (3) isolates evaluation drift on identical tables.
(2) vs (3) is a contemporaneous-code control the E42 entry never used.
"""
from __future__ import annotations
import csv, glob, json
from pathlib import Path
import numpy as np

OURS = "benedict_task4"
M = ["score", "suicides", "survived", "killed_by_opponent", "steps", "kills", "coins",
     "crates", "bombs"]


def summarise(path):
    rows = [r for r in csv.DictReader(open(path)) if r["code"] == OURS]
    out = {m: np.mean([float(r[m]) for r in rows]) for m in M}
    st = np.array([float(r["steps"]) for r in rows])
    su = np.array([int(r["suicides"]) for r in rows])
    out["sui<200"] = ((su == 1) & (st < 200)).mean()
    out["P(alive200)"] = (st >= 200).mean()
    out["haz>=200"] = ((su == 1) & (st >= 200)).sum() / max((st >= 200).sum(), 1)
    out["n"] = len(rows)
    return out


groups = {}
groups["E37 PLB2 s100-107 (as eval'd in E37)"] = [
    f"results/eval/task4_tournament/benedict_q_e37_PLB2_s{s}__ep20000__task4_rb_val550731.csv"
    for s in range(100, 108)]
groups["E37 PLB2 s108-114 (unused by E42)"] = [
    f"results/eval/task4_tournament/benedict_q_e37_PLB2_s{s}__ep20000__task4_rb_val550731.csv"
    for s in range(108, 115)]
groups["E38 s120-124 @20k (post-cleanup code)"] = [
    f"results/eval/task4_tournament/benedict_q_e38_s{s}__ep20000__task4_rb_val550731.csv"
    for s in range(120, 125)]
groups["E42 ctl s100-107 (re-eval 08-19)"] = [
    f"results/eval/task4_tournament/e42_ctl_s{s}__task4_guard_val550731.csv"
    for s in range(100, 108)]
groups["E42 mix s200-207"] = [
    f"results/eval/task4_tournament/e42_mix_s{s}__task4_guard_val550731.csv"
    for s in range(200, 208)]

cols = ["score", "suicides", "survived", "steps", "kills", "sui<200", "P(alive200)", "haz>=200"]
print(f"{'group':<42}{'seeds':>6}{'rnds':>6}" + "".join(f"{c:>13}" for c in cols))
store = {}
for g, files in groups.items():
    per = []
    for f in files:
        if not Path(f).exists():
            print("MISSING", f); continue
        per.append(summarise(f))
    if not per: continue
    store[g] = per
    n = per[0]["n"]
    print(f"{g:<42}{len(per):6d}{n:6d}" +
          "".join(f"{np.mean([p[c] for p in per]):13.3f}" for c in cols))
    print(f"{'   (between-seed sd)':<42}{'':6}{'':6}" +
          "".join(f"{np.std([p[c] for p in per], ddof=1):13.3f}" for c in cols))

print()
print("Per-seed suicides, E37 eval vs E42 re-eval of the SAME eight tables:")
a = [p["suicides"] for p in store["E37 PLB2 s100-107 (as eval'd in E37)"]]
b = [p["suicides"] for p in store["E42 ctl s100-107 (re-eval 08-19)"]]
for s, x, y in zip(range(100, 108), a, b):
    print(f"   s{s}  E37 {x:.3f}   E42 {y:.3f}   d {y-x:+.3f}")
print(f"   mean diff {np.mean(b)-np.mean(a):+.4f}")

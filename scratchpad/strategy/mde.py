"""Between-seed SD of each rung-4 metric -> the MDE any pre-registration must state.

`benedict_task4.md` §5.3 measured that run-level pairing buys nothing (corr(arm,
control) at matched seed runs -0.47..+0.44), so the honest planning quantity is
the between-seed SD of an arm mean, not a paired SD. This reports both.
"""
from __future__ import annotations
import csv, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[2] / "results/eval/task4_tournament"
US = "benedict_task4"
MET = ("score", "coins", "kills", "won", "survived", "suicides", "crates", "bombs")

def seed_means(pattern):
    out = {}
    for f in sorted(glob.glob(str(D / pattern))):
        m = re.search(r"_s(\d+)__", f)
        if not m: continue
        rows = [r for r in csv.DictReader(open(f)) if r["code"] == US]
        if not rows: continue
        out[int(m.group(1))] = {k: np.mean([float(r[k]) for r in rows]) for k in MET}
    return out

ARMS = {
    "e33_ctl":  "benedict_q_e33_ctl_s10?__ep20000__task4_rb_val550731.csv",
    "e33_F005": "benedict_q_e33_F005_s10?__ep20000__task4_rb_val550731.csv",
    "e33_F020": "benedict_q_e33_F020_s10?__ep20000__task4_rb_val550731.csv",
    "e33_F080": "benedict_q_e33_F080_s10?__ep20000__task4_rb_val550731.csv",
    "e36_OPP":  "benedict_q_e36_OPP_s10?__ep20000__task4_rb_val550731.csv",
    "e36_PLB":  "benedict_q_e36_PLB_s10?__ep20000__task4_rb_val550731.csv",
}
sm = {k: seed_means(v) for k, v in ARMS.items()}
print(f"{'metric':<10}{'pooled within-arm SD across seeds':>36}"
      f"{'  MDE n=5':>10}{'  n=15':>8}{'  n=25':>8}")
for k in MET:
    sds = []
    for arm, d in sm.items():
        v = np.array([d[s][k] for s in sorted(d)])
        if len(v) > 1: sds.append(v.std(ddof=1))
    sd = float(np.sqrt(np.mean(np.square(sds))))
    # two-arm comparison, 80% power, alpha 0.05 two-sided: 2.8 * sd * sqrt(2/n)
    f = lambda n: 2.8 * sd * np.sqrt(2 / n)
    print(f"{k:<10}{sd:>36.4f}{f(5):>10.3f}{f(15):>8.3f}{f(25):>8.3f}")

print("\nper-seed arm means (ep20000, val seed 550731):")
for arm, d in sm.items():
    v = np.array([d[s]["score"] for s in sorted(d)])
    kv = np.array([d[s]["kills"] for s in sorted(d)])
    print(f"  {arm:<9} score {np.round(v,3)}  mean {v.mean():.3f}")
    print(f"  {'':<9} kills {np.round(kv,3)}  mean {kv.mean():.3f}")

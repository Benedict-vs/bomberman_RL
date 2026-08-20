"""Audit 12 / A7: how much does guard-field suicide rate vary among tables trained on
the SAME field? E37 gives 4 arms x 15 seeds, all 20k episodes vs 3x rule_based, all
evaluated at eps=0 vs 3x rule_based on val550731. Nothing about the training FIELD
differs across those 60 tables.
"""
from __future__ import annotations
import csv, glob
import numpy as np
OURS = "benedict_task4"

def summ(p):
    R = [r for r in csv.DictReader(open(p)) if r["code"] == OURS]
    st = np.array([float(r["steps"]) for r in R]); su = np.array([int(r["suicides"]) for r in R])
    return (np.mean([float(r["score"]) for r in R]), su.mean(),
            np.mean([int(r["survived"]) for r in R]),
            ((su == 1) & (st >= 200)).sum() / max((st >= 200).sum(), 1))

print(f"{'arm':<8}{'n':>4}{'score':>9}{'suicides':>10}{'sd':>7}{'min':>7}{'max':>7}"
      f"{'survived':>10}{'haz>=200':>10}")
for arm in ("ctl2", "PLB2", "PAR", "SHF"):
    fs = sorted(glob.glob(f"results/eval/task4_tournament/benedict_q_e37_{arm}_s*__ep20000__task4_rb_val550731.csv"))
    v = np.array([summ(f) for f in fs])
    print(f"{arm:<8}{len(fs):4d}{v[:,0].mean():9.3f}{v[:,1].mean():10.3f}{v[:,1].std(ddof=1):7.3f}"
          f"{v[:,1].min():7.3f}{v[:,1].max():7.3f}{v[:,2].mean():10.3f}{v[:,3].mean():10.3f}")
print("\nE42 mix (trained on the mixed field, same map as PLB2):")
v = np.array([summ(f"results/eval/task4_tournament/e42_mix_s{s}__task4_guard_val550731.csv")
              for s in range(200, 208)])
print(f"{'mix':<8}{8:4d}{v[:,0].mean():9.3f}{v[:,1].mean():10.3f}{v[:,1].std(ddof=1):7.3f}"
      f"{v[:,1].min():7.3f}{v[:,1].max():7.3f}{v[:,2].mean():10.3f}{v[:,3].mean():10.3f}")

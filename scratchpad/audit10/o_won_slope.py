#!/usr/bin/env python3
"""Audit 10 -- the +0.088 won-per-point slope is a partial derivative in OUR score with the
opponents held fixed (survival_value.py `slope()` adds d to `score` and leaves `best` alone).
AGENTS.md now states it as 'won is a linear readout of score', which invites using it to
predict won from a measured score change. The hunt ceiling is the counterexample."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

CEIL = Path("/Users/benedictvonschubert/Projects/bomberman_RL/scratchpad/strategy/ceil")


def per_round(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(int(r["round"]), []).append(r)
    ks = sorted(by)
    mine = np.array([float([r for r in by[k] if r["code"] == "user_agent"][0]["score"]) for k in ks])
    won = np.array([float([r for r in by[k] if r["code"] == "user_agent"][0]["won"]) for k in ks])
    best = np.array([max(float(r["score"]) for r in by[k] if r["code"] != "user_agent") for k in ks])
    return mine, won, best


ctl_s, ctl_w, ctl_b = per_round(CEIL / "huntceil4k_k-1__task4_rb_ship990731.csv")
k4_s, k4_w, k4_b = per_round(CEIL / "huntceil4k_k4__task4_rb_ship990731.csv")

m = ctl_s - ctl_b
print(f"won == (our score >= best opponent)?  {(ctl_w == (m >= 0)).mean():.1%} of rounds")
for d in (0.116, 0.5, 1.0):
    print(f"  survival_value slope applied to the control margins, d=+{d}: "
          f"dP(won) = {((m + d >= 0).mean() - (m >= 0).mean()):+.4f}")
print(f"\nactually measured for the k=4 oracle: dscore {k4_s.mean()-ctl_s.mean():+.4f}, "
      f"dwon {k4_w.mean()-ctl_w.mean():+.4f}")
print(f"predicted from '0.088 per score point':                 dwon "
      f"{0.088*(k4_s.mean()-ctl_s.mean()):+.4f}  -> understates by "
      f"{(k4_w.mean()-ctl_w.mean())/(0.088*(k4_s.mean()-ctl_s.mean())):.1f}x")
dm = (k4_s - k4_b).mean() - m.mean()
print(f"the missing term: the best opponent's score fell by {ctl_b.mean()-k4_b.mean():+.4f}, "
      f"so the margin moved {dm:+.4f}")
print(f"won per point of MARGIN, from the same slope function: "
      f"{((m + dm >= 0).mean() - (m >= 0).mean())/dm:+.4f} per point "
      f"-> predicted dwon {((m + dm >= 0).mean() - (m >= 0).mean()):+.4f}")

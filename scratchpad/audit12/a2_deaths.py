"""Audit 12 / A2: WHEN the extra guard-field suicides happen, and what they cost.

E42 reads suicides +0.183 / survived -0.170 on the guard field as 'the agent stopped
paying to avoid death'. If that is a devaluation of death it should be visible as
earlier deaths across the whole round. If instead the extra deaths land after the
round's economy has closed (~step 200, scratchpad/strategy/), they cost almost nothing,
which is what score -0.049 already says.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"


def rows(arm, seeds, field):
    out = []
    for s in seeds:
        f = EVAL / f"e42_{arm}_s{s}__task4_{field}_val550731.csv"
        for r in csv.DictReader(open(f)):
            if r["code"] == OURS:
                out.append(r)
    return out


for field in ("guard", "indist", "heldout"):
    print("=" * 90)
    print("FIELD", field)
    for arm, seeds in (("ctl", range(100, 108)), ("mix", range(200, 208))):
        R = rows(arm, seeds, field)
        st = np.array([float(r["steps"]) for r in R])
        died = np.array([int(r["died"]) for r in R])
        sui = np.array([int(r["suicides"]) for r in R])
        kb = np.array([int(r["killed_by_opponent"]) for r in R])
        sc = np.array([float(r["score"]) for r in R])
        print(f"  {arm}: n={len(R)}  died={died.mean():.3f} sui={sui.mean():.3f} kb={kb.mean():.3f}")
        for lab, mask in (("suicide", sui == 1), ("killed_by", kb == 1), ("survived", died == 0)):
            if mask.sum():
                print(f"      {lab:<10} n={mask.sum():5d}  mean death-step={st[mask].mean():6.1f}"
                      f"  median={np.median(st[mask]):6.1f}  score={sc[mask].mean():.3f}")
        # deaths before / after step 200
        early = ((sui == 1) & (st < 200)).mean()
        late = ((sui == 1) & (st >= 200)).mean()
        print(f"      suicides/round  before step200 = {early:.3f}   at/after = {late:.3f}")
        # score conditional on surviving
        print(f"      score | survived  = {sc[died==0].mean():.3f}   | suicided = {sc[sui==1].mean():.3f}")

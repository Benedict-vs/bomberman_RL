"""Audit 11 -- was the +0.25 bar arithmetically reachable?

Reads the E40 CSVs only; no rollout. Decomposes each round into:
  - deaths among the three rule_based opponents
  - how many of those were their own suicides (never re-attributable to us)
  - how many were credited to us (our kills) vs to another opponent
The 'takeable pool' is the ceiling on extra kills any hunting policy could win
if total opponent mortality is held fixed (audit 10 F3's credit-transfer model).
"""
import csv, sys
from collections import defaultdict
import numpy as np

D = "scratchpad/benedict/e40"
OURS = "user_agent"

def load(tag):
    by = defaultdict(list)
    with open(f"{D}/e40_{tag}__task4_rb_ship990731.csv") as fh:
        for row in csv.DictReader(fh):
            by[int(row["round"])].append(row)
    return by

def stats(tag):
    by = load(tag)
    rows = sorted(by)
    out = defaultdict(list)
    for r in rows:
        rs = by[r]
        us = next(x for x in rs if x["code"] == OURS)
        op = [x for x in rs if x["code"] != OURS]
        our_kills = float(us["kills"])
        opp_deaths = sum(float(x["died"]) for x in op)
        opp_suic   = sum(float(x["suicides"]) for x in op)
        opp_kills  = sum(float(x["kills"]) for x in op)
        our_death  = float(us["died"]); our_suic = float(us["suicides"])
        # deaths of opponents not caused by their own bomb
        opp_killed_by_someone = opp_deaths - opp_suic
        out["our_kills"].append(our_kills)
        out["opp_deaths"].append(opp_deaths)
        out["opp_suicides"].append(opp_suic)
        out["opp_kills_total"].append(opp_kills)
        out["opp_deaths_by_others"].append(opp_killed_by_someone)
        out["takeable_left"].append(opp_killed_by_someone - our_kills)
        out["total_deaths"].append(opp_deaths + our_death)
        out["our_suicides"].append(our_suic)
        out["our_score"].append(float(us["score"]))
        out["steps"].append(float(us["steps"]))
        out["survived"].append(float(us["survived"]))
        out["bombs"].append(float(us["bombs"]))
    return rows, {k: np.array(v) for k, v in out.items()}

tags = ["ctl", "k4stale", "k4sim", "k8sim"]
res = {t: stats(t) for t in tags}
keys = list(res["ctl"][1])
print(f"{'metric':<24}" + "".join(f"{t:>12}" for t in tags))
for k in keys:
    print(f"{k:<24}" + "".join(f"{res[t][1][k].mean():>12.4f}" for t in tags))

print("\npaired diffs vs ctl (mean [95% normal CI], t)")
rounds_ctl = res["ctl"][0]
for t in tags[1:]:
    print(f"-- {t}")
    for k in keys:
        a = res["ctl"][1][k]; b = res[t][1][k]
        d = b - a
        se = d.std(ddof=1)/np.sqrt(len(d))
        print(f"   {k:<24} {d.mean():+.4f} [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]  t={d.mean()/se:+.2f}")

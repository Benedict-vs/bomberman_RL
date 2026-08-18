"""Audit 11 F: what does the oracle actually do to the field? (independent recompute)"""
import csv, numpy as np
from collections import defaultdict
from pathlib import Path
D = Path("scratchpad/benedict/e40")
def load(tag):
    br = defaultdict(list)
    with open(D / f"e40_{tag}__task4_rb_ship990731.csv") as fh:
        for row in csv.DictReader(fh): br[int(row["round"])].append(row)
    return br
def series(br, fn):
    return np.array([fn(br[r]) for r in sorted(br)])
def mine(rows, k): return float(next(r for r in rows if r["code"]=="user_agent")[k])
def oppsum(rows, k): return sum(float(r[k]) for r in rows if r["code"]!="user_agent")
M = {
 "our_kills":      lambda rows: mine(rows,"kills"),
 "opp_suicides":   lambda rows: oppsum(rows,"suicides"),
 "opp_kills":      lambda rows: oppsum(rows,"kills"),
 "opp_deaths":     lambda rows: oppsum(rows,"died"),
 "opp_score":      lambda rows: oppsum(rows,"score")/3.0,
 "our_score":      lambda rows: mine(rows,"score"),
 "round_steps":    lambda rows: float(rows[0]["round_steps"]),
}
ctl = load("ctl")
print(f"{'metric':14s} {'ctl':>9s}" + "".join(f"{t:>28s}" for t in ("k4stale","k4sim","k8sim")))
for name, fn in M.items():
    a = series(ctl, fn)
    line = f"{name:14s} {a.mean():9.4f}"
    for t in ("k4stale","k4sim","k8sim"):
        b = series(load(t), fn); d = b - a
        se = d.std(ddof=1)/np.sqrt(len(d))
        line += f"  {d.mean():+7.4f} [{d.mean()-1.96*se:+7.4f},{d.mean()+1.96*se:+7.4f}] t={d.mean()/se:+5.2f}"
    print(line)

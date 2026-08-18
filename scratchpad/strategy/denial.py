"""What is an opponent's *removal* worth to us, separately from the +5?

`round_economy.py` shows the whole economy (9 coins, 122 crates) is consumed by
step ~200, so the round is a race. An opponent removed early stops competing for
what is left. rule_based suicides are exogenous to us (they are its own bomb), so
conditioning on how many opponents were gone early is close to a natural
experiment -- provided we condition on our own survival, otherwise the rounds
where we die early contaminate it.

Uses each agent's `steps` column (its alive steps) as its death time.
"""
from __future__ import annotations

import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/eval/task4_tournament"
US = "benedict_task4"
NUM = ("survived","score","coins","kills","suicides","crates","bombs","steps",
       "died","killed_by_opponent","won","round_steps")

rows = []
for f in (sys.argv[1:] or ["benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv",
                           "benedict_task4_shipped_e37__task4_rb_ship990731.csv"]):
    tag = f
    with open(D / f) as fh:
        for r in csv.DictReader(fh):
            for k in NUM: r[k] = float(r[k])
            r["key"] = (tag, int(r["round"]))
            rows.append(r)

by = defaultdict(list)
for r in rows: by[r["key"]].append(r)
keys = sorted(by)
us   = [ [x for x in by[k] if x["code"]==US][0] for k in keys ]
opp  = [ [x for x in by[k] if x["code"]!=US] for k in keys ]
g = lambda key: np.array([m[key] for m in us])
our_steps = g("steps")

for cutoff in (100, 150, 200):
    gone = np.array([sum(1 for o in os_ if o["steps"] < cutoff) for os_ in opp])
    print(f"\n== opponents already dead by step {cutoff} "
          f"(restricted to rounds where WE live past 200) ==")
    live = our_steps >= 200
    print(f"{'k gone':>7}{'n':>6}{'ourScore':>10}{'ourCoins':>10}{'ourKills':>10}"
          f"{'ourCrates':>11}{'won':>8}{'bestOpp':>9}")
    best = np.array([max(o["score"] for o in os_) for os_ in opp])
    for k in sorted(set(gone.tolist())):
        m = live & (gone == k)
        if m.sum() < 15: continue
        print(f"{k:>7}{int(m.sum()):>6}{g('score')[m].mean():>10.3f}"
              f"{g('coins')[m].mean():>10.3f}{g('kills')[m].mean():>10.3f}"
              f"{g('crates')[m].mean():>11.2f}{g('won')[m].mean():>8.3f}"
              f"{best[m].mean():>9.3f}")

# the same, per additional opponent-step of competition during phase 1
comp = np.array([sum(min(o["steps"], 200) for o in os_) for os_ in opp])
live = our_steps >= 200
print(f"\ncorr(opponent phase-1 steps, our coins | we live past 200) = "
      f"{np.corrcoef(comp[live], g('coins')[live])[0,1]:+.3f}   "
      f"n={int(live.sum())}")
print(f"corr(opponent phase-1 steps, our crates | same) = "
      f"{np.corrcoef(comp[live], g('crates')[live])[0,1]:+.3f}")
# linear slope: coins per 100 opponent-steps removed
A = np.vstack([comp[live], np.ones(live.sum())]).T
sl = np.linalg.lstsq(A, g("coins")[live], rcond=None)[0][0]
print(f"slope: {100*sl:+.4f} of our coins per 100 opponent phase-1 steps "
      f"(one opponent removed at step 60 = 140 steps = {140*sl:+.3f} coins)")

# crate share
print(f"\ncrate accounting: ours {g('crates').mean():.2f}, "
      f"opponents {np.mean([sum(o['crates'] for o in os_) for os_ in opp]):.2f}, "
      f"total {g('crates').mean()+np.mean([sum(o['crates'] for o in os_) for os_ in opp]):.2f} of 122")
print(f"coins per crate we destroy: {g('coins').mean()/g('crates').mean():.4f}; "
      f"bombs {g('bombs').mean():.2f}, crates/bomb {g('crates').mean()/g('bombs').mean():.3f}")

"""Audit 11 -- E39 ran a k = 0 arm (bomb only when already standing on a trap
site, never walk). E40 dropped it. At n = 4000 it isolates the *walking* half of
the strategy Benedict actually proposed ("work one's way toward the nearest
opponent"), which is the half a feature would have to learn."""
import csv, numpy as np
from collections import defaultdict
D="scratchpad/strategy/ceil"
def load(t):
    by=defaultdict(list)
    for r in csv.DictReader(open(f"{D}/{t}__task4_rb_ship990731.csv")):
        by[int(r["round"])].append(r)
    return by
def M(rows):
    me=next(x for x in rows if x["code"]=="user_agent")
    opp=[float(x["score"]) for x in rows if x["code"]!="user_agent"]
    return dict(score=float(me["score"]),
                margin_mean=float(me["score"])-np.mean(opp),
                margin_best=float(me["score"])-max(opp),
                kills=float(me["kills"]), coins=float(me["coins"]),
                crates=float(me["crates"]), suicides=float(me["suicides"]))
A={t:load(t) for t in ("huntceil4k_k-1","huntceil4k_k0","huntceil4k_k4")}
R=sorted(set(A["huntceil4k_k-1"])&set(A["huntceil4k_k0"])&set(A["huntceil4k_k4"]))
keys=list(M(A["huntceil4k_k-1"][R[0]]))
V={t:{k:np.array([M(A[t][r])[k] for r in R]) for k in keys} for t in A}
print(f"E39 arms, n={len(R)} paired arenas, stale trap model")
print(f"{'metric':<13}{'k=-1':>9}{'k=0':>9}{'k=4':>9}   {'k0-ctl':>26}{'k4-ctl':>26}{'k0-k4':>26}")
for k in keys:
    ctl=V["huntceil4k_k-1"][k]; k0=V["huntceil4k_k0"][k]; k4=V["huntceil4k_k4"][k]
    def f(d):
        se=d.std(ddof=1)/np.sqrt(len(d))
        return f"{d.mean():+.3f}[{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]"
    print(f"{k:<13}{ctl.mean():>9.3f}{k0.mean():>9.3f}{k4.mean():>9.3f}   "
          f"{f(k0-ctl):>26}{f(k4-ctl):>26}{f(k0-k4):>26}")
print("\nbombs/round: k0 1276/4000 = 0.319 (0 walk steps) ; k4 1622/4000 = 0.406 (15733 walks = 3.93/round)")

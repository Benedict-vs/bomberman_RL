"""Audit 11 -- all four E40 arms against the control on the pre-registered metrics,
plus the k8 arm which E40 only ever compared against k4."""
import csv, numpy as np
from collections import defaultdict
D="scratchpad/benedict/e40"
def load(t):
    by=defaultdict(list)
    for r in csv.DictReader(open(f"{D}/e40_{t}__task4_rb_ship990731.csv")):
        by[int(r["round"])].append(r)
    return by
def M(rows):
    me=next(x for x in rows if x["code"]=="user_agent")
    opp=[float(x["score"]) for x in rows if x["code"]!="user_agent"]
    return dict(score=float(me["score"]),
                margin_mean=float(me["score"])-np.mean(opp),
                margin_best=float(me["score"])-max(opp),
                kills=float(me["kills"]), won=float(me["won"]),
                coins=float(me["coins"]), crates=float(me["crates"]))
A={t:load(t) for t in ("ctl","k4stale","k4sim","k8sim")}
R=sorted(A["ctl"])
V={t:{k:np.array([M(A[t][r])[k] for r in R]) for k in M(A["ctl"][R[0]])} for t in A}
print("paired vs ctl, n=8000, normal-theory 95% CI\n")
print(f"{'arm':<10}{'walks':>8}{'bombs':>7}" + "".join(f"{k:>26}" for k in ("score","margin_best","kills")))
meta={"k4stale":(31778,3364),"k4sim":(23501,1826),"k8sim":(65508,2131)}
for t in ("k4stale","k4sim","k8sim"):
    line=f"{t:<10}{meta[t][0]:>8}{meta[t][1]:>7}"
    for k in ("score","margin_best","kills"):
        d=V[t][k]-V["ctl"][k]; se=d.std(ddof=1)/np.sqrt(len(d))
        line+=f"   {d.mean():+.3f} [{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]"
    print(line)
print("\nall metrics, each arm vs ctl:")
for k in ("score","margin_mean","margin_best","kills","won","coins","crates"):
    row=f"  {k:<12}"
    for t in ("k4stale","k4sim","k8sim"):
        d=V[t][k]-V["ctl"][k]; se=d.std(ddof=1)/np.sqrt(len(d))
        row+=f"{t}:{d.mean():+.3f}(t={d.mean()/se:+.2f})  "
    print(row)
print("\ndose-response: E40 never reported k8sim vs ctl.")
for k in ("score","margin_best"):
    d=V["k8sim"][k]-V["ctl"][k]; se=d.std(ddof=1)/np.sqrt(len(d))
    print(f"  k8sim-ctl {k}: {d.mean():+.4f} [{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}] t={d.mean()/se:+.2f}")

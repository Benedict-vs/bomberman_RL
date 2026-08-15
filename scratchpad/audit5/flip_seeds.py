import csv, numpy as np, glob, os
def me(f, lim=None):
    rows=[r for r in csv.DictReader(open(f)) if r["agent"]=="benedict_task4"]
    d={int(r["round"]):r for r in rows}
    return {k:v for k,v in d.items() if lim is None or k<lim}
def boot(d,B=10000,seed=0):
    rng=np.random.default_rng(seed); n=len(d); m=d[rng.integers(0,n,(B,n))].mean(axis=1)
    return d.mean(), np.percentile(m,2.5), np.percentile(m,97.5)
M=("score","won","suicides","survived","crates","kills","killed_by_opponent")
print("single-cell flip of row 55060 -> UP, no training, 300 paired arenas at seed 550731")
for s in (80,81,83):
    b=me(f"results/eval/task4_tournament/benedict_q_e31_S0_s{s}__ep20000__task4_rb_val550731.csv",300)
    o=me(f"scratchpad/audit5/eval/a5flip_s{s}.csv")
    keys=sorted(set(b)&set(o))
    print(f"\n seed {s} (n={len(keys)})")
    for m in M:
        a=np.array([float(b[k][m]) for k in keys]); c=np.array([float(o[k][m]) for k in keys])
        d,lo,hi=boot(c-a); star="*" if (lo>0 or hi<0) else " "
        print(f"   {m:20s} {a.mean():7.3f} -> {c.mean():7.3f}  diff {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {star}")

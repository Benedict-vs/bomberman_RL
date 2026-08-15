import csv, numpy as np, glob, os
def me(f, lim=None):
    rows=[r for r in csv.DictReader(open(f)) if r["agent"]=="benedict_task4"]
    d={int(r["round"]):r for r in rows}
    return {k:v for k,v in d.items() if lim is None or k<lim}
def boot(d,B=10000,seed=0):
    rng=np.random.default_rng(seed); n=len(d); m=d[rng.integers(0,n,(B,n))].mean(axis=1)
    return d.mean(), np.percentile(m,2.5), np.percentile(m,97.5)
M=("score","won","suicides","survived","crates","coins","kills","killed_by_opponent")
for tag in ("esc","flip"):
    print(f"\n##### {tag}: pooled over the seeds available, 300 paired arenas each")
    A={},; A={}; B={}
    seeds=[]
    for f in sorted(glob.glob(f"scratchpad/audit5/eval/a5{tag}_s*.csv")):
        s=int(f.split("_s")[-1].split(".")[0]); seeds.append(s)
        b=me(f"results/eval/task4_tournament/benedict_q_e31_S0_s{s}__ep20000__task4_rb_val550731.csv",300)
        o=me(f)
        for k in sorted(set(b)&set(o)):
            A.setdefault(k,[]).append(b[k]); B.setdefault(k,[]).append(o[k])
    print(" seeds:",seeds)
    keys=sorted(A)
    for m in M:
        a=np.array([np.mean([float(r[m]) for r in A[k]]) for k in keys])
        c=np.array([np.mean([float(r[m]) for r in B[k]]) for k in keys])
        d,lo,hi=boot(c-a); star="*" if (lo>0 or hi<0) else " "
        print(f"   {m:20s} {a.mean():7.3f} -> {c.mean():7.3f}  diff {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {star}")

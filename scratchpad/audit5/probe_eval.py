import csv, glob, numpy as np, os
def me(f):
    rows=[r for r in csv.DictReader(open(f)) if r["agent"]=="benedict_task4"]
    return {int(r["round"]):r for r in rows}
def boot(d,B=20000,seed=0):
    rng=np.random.default_rng(seed);n=len(d);m=d[rng.integers(0,n,(B,n))].mean(1)
    return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
M=("score","won","suicides","survived","crates","coins","kills","killed_by_opponent","bombs","steps")
arms=("a5ctl","a5k15","a5k30")
data={}
for arm in arms:
    for ep in (5000,10000,20000):
        fs=sorted(glob.glob(f"scratchpad/audit5/eval/{arm}_s9?__ep{ep}.csv"))
        if not fs: continue
        per={}
        for f in fs:
            for k,v in me(f).items(): per.setdefault(k,[]).append(v)
        data[(arm,ep)]={m:np.array([np.mean([float(r[m]) for r in per[k]]) for k in sorted(per)]) for m in M}
        data[(arm,ep)]["_n"]=len(fs)
print(f"{'arm':>7}{'ep':>7}  "+"".join(f"{m[:9]:>10}" for m in M))
for arm in arms:
    for ep in (5000,10000,20000):
        d=data.get((arm,ep))
        if not d: continue
        print(f"{arm:>7}{ep:>7}  "+"".join(f"{d[m].mean():10.3f}" for m in M))
print("\npaired vs a5ctl at the same episode count (300 arenas, seeds averaged per arena):")
for ep in (5000,10000,20000):
    if ("a5ctl",ep) not in data: continue
    for arm in ("a5k15","a5k30"):
        if (arm,ep) not in data: continue
        print(f" ep{ep}  {arm} - a5ctl")
        for m in M:
            d,lo,hi=boot(data[(arm,ep)][m]-data[("a5ctl",ep)][m])
            star="*" if (lo>0 or hi<0) else " "
            print(f"    {m:20s} {d:+8.3f} [{lo:+.3f},{hi:+.3f}] {star}")
print("\nK30 - K15 at ep20000 (the dose-response test):")
for m in M:
    d,lo,hi=boot(data[("a5k30",20000)][m]-data[("a5k15",20000)][m])
    star="*" if (lo>0 or hi<0) else " "
    print(f"    {m:20s} {d:+8.3f} [{lo:+.3f},{hi:+.3f}] {star}")

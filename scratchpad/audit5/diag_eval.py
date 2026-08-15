import csv, glob, numpy as np, os, sys
def me(f):
    rows=[r for r in csv.DictReader(open(f)) if r["agent"]=="benedict_task4"]
    return {int(r["round"]):r for r in rows}
base={}
# baseline: first 300 rounds of the 1000-round e31 s80 ep20000 eval at the same seed
b=me("results/eval/task4_tournament/benedict_q_e31_S0_s80__ep20000__task4_rb_val550731.csv")
base={k:v for k,v in b.items() if k<300}
def boot(d,B=10000,seed=0):
    rng=np.random.default_rng(seed); n=len(d); m=d[rng.integers(0,n,(B,n))].mean(axis=1)
    return d.mean(), np.percentile(m,2.5), np.percentile(m,97.5)
M=("score","won","suicides","survived","crates","coins","kills","killed_by_opponent","bombs","steps")
print(f"{'metric':22s} {'base(e31 s80 20k)':>18s}")
for m in M:
    a=np.array([float(base[k][m]) for k in sorted(base)])
    print(f"{m:22s} {a.mean():18.3f}")
for f in sorted(glob.glob("scratchpad/audit5/eval/a5*.csv")):
    o=me(f); keys=sorted(set(base)&set(o))
    print(f"\n--- {os.path.basename(f)}  (n={len(keys)} paired arenas)")
    for m in M:
        a=np.array([float(base[k][m]) for k in keys]); c=np.array([float(o[k][m]) for k in keys])
        d,lo,hi=boot(c-a)
        star="*" if (lo>0 or hi<0) else " "
        print(f"  {m:22s} {a.mean():8.3f} -> {c.mean():8.3f}   diff {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {star}")

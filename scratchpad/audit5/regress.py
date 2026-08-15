import csv, glob, numpy as np, os
def stats(f):
    rows=[r for r in csv.DictReader(open(f)) if r["agent"]=="benedict_task4"]
    g=lambda k: np.array([float(r[k]) for r in rows])
    return {k:g(k).mean() for k in ("score","won","suicides","survived","crates","coins","kills","killed_by_opponent")}
pts=[]
for f in sorted(glob.glob("results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep*__task4_rb_val550731.csv")):
    s=stats(f); s["label"]=os.path.basename(f).split("__task4")[0]; pts.append(s)
x=np.array([p["suicides"] for p in pts]); y=np.array([p["won"] for p in pts]); z=np.array([p["score"] for p in pts])
def lin(x,y,name):
    A=np.vstack([x,np.ones_like(x)]).T
    b,res,_,_=np.linalg.lstsq(A,y,rcond=None)
    r=np.corrcoef(x,y)[0,1]
    # bootstrap slope CI
    rng=np.random.default_rng(0); sl=[]
    for _ in range(10000):
        i=rng.integers(0,len(x),len(x))
        try: sl.append(np.linalg.lstsq(np.vstack([x[i],np.ones(len(i))]).T,y[i],rcond=None)[0][0])
        except Exception: pass
    lo,hi=np.percentile(sl,[2.5,97.5])
    print(f"  {name}: slope {b[0]:+.3f} [{lo:+.3f},{hi:+.3f}]  r={r:+.3f}  (n={len(x)} run x checkpoint points)")
print("across the 15 E31 (seed x checkpoint) evaluations, 1000 rounds each:")
lin(x,y,"won ~ suicides "); lin(x,z,"score ~ suicides")
print("\nsuicides per point:", np.round(x,3))
print("won        per point:", np.round(y,3))

# reference: rule_based in its own field
for f in sorted(glob.glob("results/eval/baselines/*task4*")+glob.glob("results/eval/task4_tournament/ref_*")):
    if f.endswith(".csv"): print("\nreference file:",f)

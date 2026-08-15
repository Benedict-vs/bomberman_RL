import pickle, numpy as np, glob, os, sys
from collections import Counter
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
v=Counter(s["row"] for s in D["step_log"])
rows=np.array(list(v)); w=np.array([v[r] for r in rows],float); w/=w.sum()
print("visit-weighted action gap on a FIXED row set (E31 s80 ep20000 greedy visits)")
for pat in sys.argv[1:]:
    fs=sorted(glob.glob(pat))
    if not fs: continue
    g=[];p=[]
    for f in fs:
        q=np.load(f)[rows]; s=np.sort(q,axis=1); d=s[:,-1]-s[:,-2]
        g.append((d*w).sum()); p.append(w[d<1.0].sum())
    print(f"  {pat:60s} n={len(fs)}  gap {np.mean(g):.3f}  mass(gap<1) {np.mean(p):.3f}")

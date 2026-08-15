import pickle, numpy as np, os, sys, importlib.util
from collections import Counter
sys.path.insert(0,os.path.abspath("."))
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
visits=Counter(s["row"] for s in D["step_log"])
rows=np.array(list(visits.keys())); w=np.array([visits[r] for r in rows],dtype=float); w/=w.sum()
print("visit-weighted action gap (Q_best - Q_2nd) on the ep20000 greedy visit distribution")
for ep in (5000,10000,20000):
    gaps=[]
    for s in range(80,85):
        q=np.load(f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep{ep}.npy")[rows]
        srt=np.sort(q,axis=1)
        g=srt[:,-1]-srt[:,-2]
        gaps.append((g*w).sum())
    print(f"  ep{ep}: {np.mean(gaps):.4f}  (per seed {np.round(gaps,3)})")
    # fraction of visit mass in near-tied rows
    fr=[]
    for s in range(80,85):
        q=np.load(f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep{ep}.npy")[rows]
        srt=np.sort(q,axis=1); g=srt[:,-1]-srt[:,-2]
        fr.append(w[g<0.25].sum())
    print(f"          visit mass with gap < 0.25: {np.mean(fr):.3f}")
    fr=[]
    for s in range(80,85):
        q=np.load(f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep{ep}.npy")[rows]
        srt=np.sort(q,axis=1); g=srt[:,-1]-srt[:,-2]
        fr.append(w[g<1.0].sum())
    print(f"          visit mass with gap < 1.00: {np.mean(fr):.3f}")
# how many greedy actions change between checkpoints, weighted by visits
print("\ngreedy-action churn on visited rows (visit-weighted), per seed:")
for s in range(80,85):
    qs={ep:np.load(f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep{ep}.npy")[rows] for ep in (5000,10000,20000)}
    a={ep:qs[ep].argmax(1) for ep in qs}
    print(f"  s{s}: 5k->10k {w[a[5000]!=a[10000]].sum():.3f}   10k->20k {w[a[10000]!=a[20000]].sum():.3f}")

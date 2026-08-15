import csv, glob, numpy as np, sys
def me(f):
    with open(f) as fh: rs=list(csv.DictReader(fh))
    return {int(r["round"]):r for r in rs if r["agent"]=="benedict_task4"}
def boot(d,B=10000,seed=0):
    rng=np.random.default_rng(seed); n=len(d)
    m=d[rng.integers(0,n,(B,n))].mean(axis=1)
    return d.mean(), np.percentile(m,2.5), np.percentile(m,97.5)
def cmp(pa,pb,metrics=("score","won","suicides","survived","crates","coins","kills","killed_by_opponent")):
    A={}; B={}
    for f in sorted(glob.glob(pa)): 
        for k,v in me(f).items(): A.setdefault(k,[]).append(v)
    for f in sorted(glob.glob(pb)):
        for k,v in me(f).items(): B.setdefault(k,[]).append(v)
    keys=sorted(set(A)&set(B))
    for m in metrics:
        a=np.array([np.mean([float(r[m]) for r in A[k]]) for k in keys])
        b=np.array([np.mean([float(r[m]) for r in B[k]]) for k in keys])
        d,lo,hi=boot(b-a)
        star="*" if (lo>0 or hi<0) else " "
        print(f"{m:20s} A={a.mean():7.3f} B={b.mean():7.3f}  diff {d:+7.3f} [{lo:+.3f},{hi:+.3f}] {star}")
if __name__=="__main__":
    print("A = ep5000 (5 seeds averaged per arena), B = ep20000")
    cmp("results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep5000__*.csv",
        "results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep20000__*.csv")
    print("\nA = ep10000, B = ep20000")
    cmp("results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep10000__*.csv",
        "results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep20000__*.csv")

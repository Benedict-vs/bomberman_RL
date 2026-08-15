"""Is Q(55060, DOWN) above its own Bellman target under the current policy?

Rewards are not recorded per step, so the target is computed with r = 0 on every
non-terminal transition (BM_STEP_COST=0; the only other live rewards are
CRATE_DESTROYED, INVALID_ACTION and WAITED, all omitted -> the target below is an
UPPER bound only up to the crate term, which is measured separately at 0.67 per
exploding bomb and applies to every action in the row alike).
Terminal transitions use r = -5 + (crates the killing bomb destroyed).
"""
import pickle, numpy as np
from collections import defaultdict
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
q=np.load("checkpoints/benedict_task4/q_table_e31_S0_s80__ep20000.npy")
G=0.99
by=defaultdict(list)
for s in D["step_log"]: by[s["round"]].append(s)
deaths={d["round"]:d for d in D["deaths"]}
for ROW,ACT in ((55060,"DOWN"),(59160,"DOWN"),(35032,"UP")):
    tg=[]; nterm=0
    for r,v in by.items():
        v.sort(key=lambda z:z["step"])
        for i,s in enumerate(v):
            if s["row"]!=ROW or s["action"]!=ACT: continue
            if i+1 < len(v):
                tg.append(G*q[v[i+1]["row"]].max())
            else:
                d=deaths.get(r)
                if d is None: continue           # survived to step 400
                nterm+=1
                arena=d["window"][-1]["arena"] if d["window"] else None
                cr=0
                if arena is not None:
                    blast=set()
                    for k in d["killers"]:
                        if k[0]=="benedict_task4": blast|={tuple(int(x) for x in t) for t in k[2]}
                    cr=sum(1 for t in blast if arena[t]==1)
                tg.append(-5.0+cr)
    tg=np.array(tg)
    a=["UP","RIGHT","DOWN","LEFT","WAIT","BOMB"].index(ACT)
    print(f"row {ROW} act {ACT}: n={len(tg)} transitions ({nterm} terminal)  "
          f"Q={q[ROW,a]:.3f}  empirical Bellman target={tg.mean():.3f}  residual={tg.mean()-q[ROW,a]:+.3f}")

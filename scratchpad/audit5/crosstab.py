import pickle
from collections import Counter
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
DEL={"UP":(0,-1),"RIGHT":(1,0),"DOWN":(0,1),"LEFT":(-1,0)}
ct=Counter()
for d in D["deaths"]:
    w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
    if ds not in w: continue
    s=w[ds]; pos=tuple(int(v) for v in s["self"]); a=s["my_action"]
    blast=set()
    for k in d["killers"]: blast|={tuple(int(v) for v in t) for t in k[2]}
    if a not in DEL: cause="WAIT/BOMB"
    else:
        tgt=(pos[0]+DEL[a][0],pos[1]+DEL[a][1])
        cause="stepped into blast" if tgt in blast else "blocked by opponent"
    c1=s["cand"][0]
    ct[(cause, f"C1_escape_timed={c1}")]+=1
for k,v in sorted(ct.items()): print(f"  {v:4d}  {k[0]:22s} {k[1]}")

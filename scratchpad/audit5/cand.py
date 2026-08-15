import pickle, numpy as np, sys
from collections import Counter
sys.path.insert(0,"scratchpad/deaths")
NAMES=["C1_escape_timed","C2_bomb_safe","C3_opp_dist","C4_exits","C5_enemy_threat","C6_trap","C7_safe_dist","C8_blocked_by_agent","C9_no_bomb"]
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
steps=D["step_log"]; N=len(steps)
base=[Counter() for _ in NAMES]
for s in steps:
    for i,v in enumerate(s["cand"]): base[i][v]+=1
death=[Counter() for _ in NAMES]
nd=0
for d in D["deaths"]:
    w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
    if ds not in w: continue
    nd+=1
    for i,v in enumerate(w[ds]["cand"]): death[i][v]+=1
print(f"base n={N} alive steps, deaths n={nd}\n")
for i,nm in enumerate(NAMES):
    vals=sorted(set(base[i])|set(death[i]))
    line=[]
    for v in vals:
        b=base[i][v]/N; dd=death[i][v]/nd
        line.append(f"{v}: base {b:.3f} death {dd:.3f} lift {dd/max(b,1e-9):5.1f}")
    print(f"{nm:22s} "+" | ".join(line))

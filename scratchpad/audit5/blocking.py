import pickle, numpy as np
from collections import Counter
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
DEL={"UP":(0,-1),"RIGHT":(1,0),"DOWN":(0,1),"LEFT":(-1,0)}
c=Counter(); n=0
for d in D["deaths"]:
    w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
    if ds not in w: continue
    n+=1
    s=w[ds]; pos=tuple(int(v) for v in s["self"]); a=s["my_action"]
    blast=set()
    for k in d["killers"]: blast|={tuple(int(v) for v in t) for t in k[2]}
    if a not in DEL:
        c["action was WAIT/BOMB"]+=1; continue
    tgt=(pos[0]+DEL[a][0], pos[1]+DEL[a][1])
    if tgt in blast:
        c["moved into the blast (target lethal)"]+=1; continue
    # target was safe -> move must have been blocked
    arena=s["arena"]
    if arena[tgt]!=0: c["target was a wall/crate (feature said blocked)"]+=1; continue
    blocked_by=None
    for name,opos,_ in s["others"]:
        opos=tuple(int(v) for v in opos)
        if opos==tgt: blocked_by="opponent already standing there"; break
        act=s["actions"].get(name)
        if act in DEL and (opos[0]+DEL[act][0],opos[1]+DEL[act][1])==tgt:
            blocked_by="opponent moved in this step"; break
    if any(tuple(int(v) for v in b[0])==tgt for b in s["bombs"]): blocked_by="bomb on the tile"
    c[f"safe target, move failed: {blocked_by}"]+=1
print(f"deaths analysed {n}")
for k,v in c.most_common(): print(f"  {v:4d} ({v/n:.3f})  {k}")
# base rate: how often is a neighbour 'clear' but an opponent takes it that step?
steps=D["step_log"]
print("\n--- how common is the hazard the feature map cannot see? ---")
print("(needs positions; computed from the death windows only, so no base rate here)")

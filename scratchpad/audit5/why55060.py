import pickle, numpy as np
from collections import Counter
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
DEL={"UP":(0,-1),"RIGHT":(1,0),"DOWN":(0,1),"LEFT":(-1,0)}
res=Counter(); oppact=Counter(); oppdist=Counter()
for d in D["deaths"]:
    w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
    if ds not in w: continue
    s=w[ds]
    if s["row"]!=55060 or s["my_action"]!="DOWN": continue
    pos=tuple(int(v) for v in s["self"])
    blast=set()
    for k in d["killers"]: blast|={tuple(int(v) for v in t) for t in k[2]}
    tgt=(pos[0],pos[1]+1)
    res[("me_in_blast",pos in blast,"target_in_blast",tgt in blast)]+=1
    # opponents at start of step d and their action
    for name,opos,_ in s["others"]:
        opos=tuple(int(v) for v in opos)
        dist=abs(opos[0]-pos[0])+abs(opos[1]-pos[1])
        a=s["actions"].get(name)
        if dist<=2:
            oppact[(dist,a)]+=1
            if a in DEL:
                np_=(opos[0]+DEL[a][0], opos[1]+DEL[a][1])
                if np_==tgt: oppdist["opponent moved INTO my target tile"]+=1
                elif opos==tgt: oppdist["opponent already ON my target tile"]+=1
        oppdist[f"min_dist_bucket_{min(dist,9)}"]+=0
    md=min(abs(int(o[1][0])-pos[0])+abs(int(o[1][1])-pos[1]) for o in s["others"]) if s["others"] else 99
    oppdist[f"nearest_opp_dist={md}"]+=1
print("blast membership:",dict(res))
print("\nnearest opponent distance at the death step:", dict(sorted((k,v) for k,v in oppdist.items() if k.startswith("nearest"))))
print("blocking:", {k:v for k,v in oppdist.items() if "target" in k})
print("\nnearby opponent (dist<=2) actions:", oppact.most_common(10))

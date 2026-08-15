import pickle, numpy as np, sys
from collections import Counter
DEL={"UP":(0,-1),"RIGHT":(1,0),"DOWN":(0,1),"LEFT":(-1,0)}
IDX={"UP":0,"RIGHT":1,"DOWN":2,"LEFT":3}
NB={0:"blocked",1:"lethal",2:"in-blast",3:"clear"}
for pkl in sys.argv[1:]:
    D=pickle.load(open(pkl,"rb"))
    c=Counter(); n=0; esc=Counter()
    for d in D["deaths"]:
        w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
        if ds not in w: continue
        n+=1
        s=w[ds]; pos=tuple(int(v) for v in s["self"]); a=s["my_action"]; dg=s["digits"]
        blast=set()
        for k in d["killers"]: blast|={tuple(int(v) for v in t) for t in k[2]}
        if a not in DEL: c["WAIT/BOMB"]+=1; continue
        tgt=(pos[0]+DEL[a][0], pos[1]+DEL[a][1])
        nbd=NB[dg[IDX[a]]]
        if tgt in blast: c[f"stepped into blast (its digit said {nbd})"]+=1
        else:
            occ=any(tuple(int(v) for v in o[1])==tgt for o in s["others"])
            mv=any(s["actions"].get(o[0]) in DEL and (int(o[1][0])+DEL[s['actions'][o[0]]][0],int(o[1][1])+DEL[s['actions'][o[0]]][1])==tgt for o in s["others"])
            c[f"blocked ({'standing' if occ else 'moved in' if mv else '?'}), its digit said {nbd}"]+=1
        # did it follow the escape digit?
        esc[(dg[5]!=0, a==(list(DEL)[dg[5]-1] if dg[5] else None))]+=1
    # base rate of the invisible hazard over all alive steps: opponent at manhattan dist 2 adjacent-through
    print(f"\n=== {pkl}: {n} deaths / {D['n_rounds']} rounds")
    for k,v in c.most_common(): print(f"  {v:4d} ({v/n:.3f})  {k}")
    print("  (escape digit valid?, action==escape dir?):", dict(esc))

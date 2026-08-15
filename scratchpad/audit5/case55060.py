import pickle, sys, numpy as np
from collections import Counter
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
cases=[]
for d in D["deaths"]:
    w={s["step"]:s for s in d["window"]}
    ds=d["death_step"]
    if ds in w and w[ds]["row"]==55060 and w[ds]["my_action"]=="DOWN":
        cases.append((d,w))
print("cases:",len(cases))
for d,w in cases[:3]:
    ds=d["death_step"]
    print(f"\n===== round {d['round']} death_step {ds} suicide={d['suicide']} killers={d['killers']}")
    for st in sorted(w):
        if st< ds-5: continue
        s=w[st]
        print(f"  step {st:3d} pos {s['self']} act {s['my_action']:6s} dig {s['digits']} bombs {s['bombs']} expl {[(e[1],e[2],e[3]) for e in s['explosions']]} others {[(o[0][-1],o[1]) for o in s['others']]}")
# aggregate: previous action / previous row
prev=Counter(); prevpos=Counter()
for d,w in cases:
    ds=d["death_step"]
    if ds-1 in w:
        a=w[ds-1]
        prev[(a["row"],a["my_action"])]+=1
        prevpos[(a["self"], w[ds]["self"])]+=1
print("\nprevious (row,action) at d-1:", prev.most_common(6))
# where does the agent end up at death? compare pos at d with killers' blast
oncenter=0
for d,w in cases:
    ds=d["death_step"]; s=w[ds]
    for k in d["killers"]:
        owner, bombtile, blast, kstep = k
        print_once=None
    # position at d
print("\npositions at d (first 10):", [w[d['death_step']]['self'] for d,w in cases[:10]])

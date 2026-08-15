import pickle, numpy as np
from collections import Counter
for pkl in ["scratchpad/audit5/deaths_e31_s80_ep20000.pkl","scratchpad/audit5/deaths_e31_s80_ep5000.pkl"]:
    D=pickle.load(open(pkl,"rb"))
    cr=[]; n=0
    for d in D["deaths"]:
        w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
        if ds not in w or not d["suicide"]: continue
        s=w[ds]; arena=s["arena"]; n+=1
        blast=set()
        for k in d["killers"]:
            if k[0]!="benedict_task4": continue
            blast|={tuple(int(v) for v in t) for t in k[2]}
        cr.append(sum(1 for t in blast if arena[t]==1))
    cr=np.array(cr)
    print(f"{pkl.split('/')[-1]}: {n} suicides; crates destroyed by the KILLING bomb: "
          f"mean {cr.mean():.2f} median {np.median(cr):.0f} dist {dict(sorted(Counter(cr.tolist()).items()))}")
    print(f"   -> terminal reward on a suicide at BM_CRATE=1.0, GOT_KILLED=-5: "
          f"mean {-5+cr.mean():.2f}  (fraction of suicides with terminal reward >= 0: {(cr>=5).mean():.3f})")

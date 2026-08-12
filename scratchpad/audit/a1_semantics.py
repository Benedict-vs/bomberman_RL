import glob, numpy as np
from lib import load
print(f"{'run':<12}{'eps':>7}{'KS':>7}{'GK':>7}{'KS&noGK':>9}{'GK&noKS':>9}{'both':>7}{'GK<KS':>7}{'surv':>7}")
for f in sorted(glob.glob("/Users/benedictvonschubert/Projects/bomberman_RL/results/train/task3_opponents/*.csv")):
    d = load(f)
    ks, gk = d["KILLED_SELF"], d["GOT_KILLED"]
    run = f.split("__")[-1][:-4]
    print(f"{run:<12}{len(ks):>7}{int(ks.sum()):>7}{int(gk.sum()):>7}"
          f"{int(((ks>0)&(gk==0)).sum()):>9}{int(((gk>0)&(ks==0)).sum()):>9}"
          f"{int(((gk>0)&(ks>0)).sum()):>7}{int((gk<ks).sum()):>7}{d['SURVIVED_ROUND'].mean():>7.3f}")

import csv, glob, numpy as np
files=sorted(glob.glob("results/train/task4_tournament/benedict_task3__q_e31_S0_s8?.csv"))
cols=None; data=[]
for f in files:
    with open(f) as fh:
        r=csv.DictReader(fh); rows=list(r)
    cols=rows[0].keys()
    data.append(rows)
print("n files",len(files),"rows/file",[len(d) for d in data])
keys=["score","steps","reward","KILLED_SELF","GOT_KILLED","SURVIVED_ROUND","CRATE_DESTROYED","COIN_COLLECTED","BOMB_DROPPED","INVALID_ACTION","WAITED","epsilon","td_error"]
BIN=1000
print(f"{'ep':>6} "+" ".join(f"{k[:9]:>9}" for k in keys))
nb=min(len(d) for d in data)//BIN
for b in range(nb):
    vals=[]
    for k in keys:
        a=np.concatenate([np.array([float(row[k]) for row in d[b*BIN:(b+1)*BIN]]) for d in data])
        vals.append(a.mean())
    print(f"{(b+1)*BIN:>6} "+" ".join(f"{v:9.3f}" for v in vals))

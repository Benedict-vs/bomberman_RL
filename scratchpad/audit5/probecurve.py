import csv, glob, numpy as np, sys
BIN=int(sys.argv[1]) if len(sys.argv)>1 else 2000
keys=["score","steps","KILLED_SELF","GOT_KILLED","SURVIVED_ROUND","CRATE_DESTROYED","COIN_COLLECTED","BOMB_DROPPED","INVALID_ACTION","WAITED","KILLED_OPPONENT"]
arms={}
for f in sorted(glob.glob("results/train/task4_tournament/benedict_task3__q_e31_a5*.csv")):
    arm=f.split("_e31_")[1].split("_s")[0]
    arms.setdefault(arm,[]).append(list(csv.DictReader(open(f))))
n=min(len(d) for v in arms.values() for d in v)
print(f"episodes available: {n}")
print(f"{'arm':>7} {'ep':>6} "+" ".join(f"{k[:9]:>9}" for k in keys))
for arm in sorted(arms):
    for b in range(n//BIN):
        vals=[]
        for k in keys:
            a=np.concatenate([np.array([float(r[k]) for r in d[b*BIN:(b+1)*BIN]]) for d in arms[arm]])
            vals.append(a.mean())
        print(f"{arm:>7} {(b+1)*BIN:>6} "+" ".join(f"{v:9.3f}" for v in vals))
    print()

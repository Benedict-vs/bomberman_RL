import sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scratchpad"/"audit"))
from lib import load
BLOCKS=[(1,500),(500,1000),(1000,2000),(2000,3200)]
print(f"{'arm':<10} {'block':>11} {'eps':>5} {'steps':>6} {'crates':>7} {'cr/st':>7} {'bombs':>6} {'coins':>6} {'inval':>6} {'suic':>5} {'kbyop':>5} {'surv':>5} {'kills':>5}")
for f in sorted(glob.glob(str(ROOT/"results/train/task3_opponents/*audit2*.csv"))):
    arm = f.split("audit2_")[1].split("_s40")[0]
    d = load(f)
    for lo,hi in BLOCKS:
        m=(d['episode']>=lo)&(d['episode']<hi)
        if m.sum()<50: continue
        g={k:float(np.nanmean(d[k][m])) for k in ['epsilon','steps','CRATE_DESTROYED','BOMB_DROPPED','COIN_COLLECTED','INVALID_ACTION','KILLED_SELF','GOT_KILLED','SURVIVED_ROUND','KILLED_OPPONENT']}
        print(f"{arm:<10} {lo:>5}-{hi:<5} {g['epsilon']:5.3f} {g['steps']:6.1f} {g['CRATE_DESTROYED']:7.2f} {g['CRATE_DESTROYED']/g['steps']:7.4f} {g['BOMB_DROPPED']:6.2f} {g['COIN_COLLECTED']:6.2f} {g['INVALID_ACTION']:6.2f} {g['KILLED_SELF']:5.3f} {g['GOT_KILLED']-g['KILLED_SELF']:5.3f} {g['SURVIVED_ROUND']:5.3f} {g['KILLED_OPPONENT']:5.3f}")
    print()

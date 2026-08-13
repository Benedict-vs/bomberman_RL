import sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scratchpad"/"audit"))
from lib import load
for arm in ["base","crate1"]:
    fs = sorted(glob.glob(str(ROOT/f"results/train/task3_opponents/*audit2b_{arm}_s4*.csv")))
    print(f"--- {arm} (n={len(fs)}) ---")
    for lo,hi in [(1,1000),(1000,2000),(2000,4000),(4000,6001)]:
        acc={}
        for f in fs:
            d=load(f); m=(d['episode']>=lo)&(d['episode']<hi)
            for k in ['epsilon','steps','CRATE_DESTROYED','BOMB_DROPPED','COIN_COLLECTED','INVALID_ACTION','KILLED_SELF','GOT_KILLED','SURVIVED_ROUND','KILLED_OPPONENT']:
                acc.setdefault(k,[]).append(float(np.nanmean(d[k][m])))
        g={k:np.mean(v) for k,v in acc.items()}
        print(f"  {lo:>5}-{hi:<5} eps {g['epsilon']:.3f} steps {g['steps']:6.1f} crates {g['CRATE_DESTROYED']:6.2f} "
              f"cr/st {g['CRATE_DESTROYED']/g['steps']:.4f} bombs {g['BOMB_DROPPED']:6.2f} coins {g['COIN_COLLECTED']:5.2f} "
              f"suic {g['KILLED_SELF']:.3f} surv {g['SURVIVED_ROUND']:.3f} kills {g['KILLED_OPPONENT']:.3f}")

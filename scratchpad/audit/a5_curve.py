import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lib import load
R = "/Users/benedictvonschubert/Projects/bomberman_RL/results/train/task3_opponents/"
BINS = [(0,50),(50,200),(200,500),(500,1000),(1000,2000),(2000,5000),(5000,10000),
        (10000,20000),(20000,30000),(30000,40000)]
for arm in "AB":
    ds = [load(f) for f in sorted(glob.glob(R + f"benedict_task3__q_e25_{arm}_s1?.csv"))]
    print(f"\n=== arm {arm}  (mean over {len(ds)} runs) ===")
    print(f"{'episodes':<14}{'eps':>6}{'steps':>8}{'surv':>7}{'suic':>7}{'kby':>7}"
          f"{'crates':>8}{'bombs':>8}{'inval':>8}{'wait':>8}{'coins':>7}{'reward':>9}{'|TD|':>8}")
    for lo, hi in BINS:
        row = {}
        for c in ["epsilon","steps","SURVIVED_ROUND","KILLED_SELF","GOT_KILLED",
                  "CRATE_DESTROYED","BOMB_DROPPED","INVALID_ACTION","WAITED",
                  "COIN_COLLECTED","reward","td_error"]:
            row[c] = np.mean([d[c][lo:hi].mean() for d in ds])
        print(f"{f'{lo}-{hi}':<14}{row['epsilon']:>6.3f}{row['steps']:>8.1f}"
              f"{row['SURVIVED_ROUND']:>7.3f}{row['KILLED_SELF']:>7.3f}"
              f"{row['GOT_KILLED']-row['KILLED_SELF']:>7.3f}"
              f"{row['CRATE_DESTROYED']:>8.2f}{row['BOMB_DROPPED']:>8.2f}"
              f"{row['INVALID_ACTION']:>8.2f}{row['WAITED']:>8.2f}{row['COIN_COLLECTED']:>7.3f}"
              f"{row['reward']:>9.2f}{row['td_error']:>8.3f}")

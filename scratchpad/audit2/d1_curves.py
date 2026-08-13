"""Training curves per arm, averaged over the 5 runs, in episode blocks."""
import sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scratchpad"/"audit"))
from lib import load

BLOCKS = [(1,200),(200,500),(500,1000),(1000,2000),(2000,5000),(5000,10000),
          (10000,20000),(20000,30000),(30000,40000)]

def arm(pat):
    files = sorted(glob.glob(str(ROOT/"results/train/task3_opponents"/pat)))
    return [load(f) for f in files]

for name, pat in [("E25 A (rung2 rewards, no feat)","*e25_A_*.csv"),
                  ("E25 B (+GOT_KILLED -10/-5)","*e25_B_*.csv"),
                  ("E26 S (fixed sem, no feat)","*e26_S_*.csv"),
                  ("E26 H (fixed sem + HUNT + kill25)","*e26_H_*.csv")]:
    runs = arm(pat)
    print(f"\n=== {name}  (n={len(runs)} runs) ===")
    hdr = f"{'block':>13} {'eps':>5} {'steps':>6} {'crates':>7} {'cr/step':>8} {'bombs':>6} {'coins':>6} {'inval':>7} {'inv/st':>7} {'suic':>5} {'kbyop':>5} {'surv':>5} {'kills':>5} {'rew':>8} {'td':>7}"
    print(hdr)
    for lo,hi in BLOCKS:
        acc = {}
        for d in runs:
            m = (d['episode']>=lo)&(d['episode']<hi)
            if m.sum()==0: continue
            for k in ['epsilon','steps','CRATE_DESTROYED','BOMB_DROPPED','COIN_COLLECTED',
                      'INVALID_ACTION','KILLED_SELF','GOT_KILLED','SURVIVED_ROUND',
                      'KILLED_OPPONENT','reward','td_error']:
                acc.setdefault(k,[]).append(np.nanmean(d[k][m]))
        if not acc: continue
        g = {k: np.mean(v) for k,v in acc.items()}
        kb = g['GOT_KILLED']-g['KILLED_SELF']
        print(f"{lo:>6}-{hi:<6} {g['epsilon']:5.3f} {g['steps']:6.1f} {g['CRATE_DESTROYED']:7.2f} "
              f"{g['CRATE_DESTROYED']/g['steps']:8.4f} {g['BOMB_DROPPED']:6.2f} {g['COIN_COLLECTED']:6.2f} "
              f"{g['INVALID_ACTION']:7.2f} {g['INVALID_ACTION']/g['steps']:7.4f} {g['KILLED_SELF']:5.3f} {kb:5.3f} "
              f"{g['SURVIVED_ROUND']:5.3f} {g['KILLED_OPPONENT']:5.3f} {g['reward']:8.2f} {g['td_error']:7.4f}")

"""Recompute E26's headline numbers from the committed evaluation CSVs."""
import sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scratchpad"/"audit"))
from lib import load, mean_ci, t_ci
E = ROOT/"results/eval/task3_opponents"

def mine(p):
    d = load(str(p))
    m = d['agent'] == 'benedict_task3'
    return {k: v[m] for k, v in d.items()}

print("== E26 ship-seed 990731, 1000 rounds, frozen table, HUNT off vs on ==")
off = mine(E/"benedict_q_e26_F_huntoff__task3_cc_ship990731.csv")
on  = mine(E/"benedict_q_e26_F_frozen__task3_cc_ship990731.csv")
assert np.array_equal(off['seed'], on['seed']), "not arena-paired"
for k in ['score','coins','kills','crates','invalid','survived','suicides']:
    print(f"  {k:<9} off {off[k].mean():8.3f}   on {on[k].mean():8.3f}")
d = on['score'] - off['score']
m, lo, hi = mean_ci(d)
print(f"  paired score diff  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(d)}")

print("\n== E26 val-seed 550731, 300 rounds: arm F vs trained arms (primary ckpt 20000) ==")
f = mine(E/"benedict_q_e26_F_frozen__task3_cc_val550731.csv")
print(f"  F  score {f['score'].mean():.3f}  n={len(f)}")
for arm in ["S", "H"]:
    per_run = []
    for s in range(20, 25):
        d = mine(E/f"benedict_q_e26_{arm}_s{s}__ep20000__task3_cc_val550731.csv")
        per_run.append(d['score'].mean())
    m, lo, hi, sd = t_ci(np.array(per_run))
    print(f"  {arm}@20k per-run scores {['%.3f'%x for x in per_run]}  mean {m:.3f} [{lo:.3f},{hi:.3f}] sd {sd:.3f}")
    diff = np.array(per_run) - f['score'].mean()
    m2, lo2, hi2, _ = t_ci(diff)
    print(f"      minus F(4.160 fixed): {m2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")

print("\n== per-round paired diff H@20k(pooled 5 runs) - F, the way E26 reported -1.688 ==")
allH = np.concatenate([mine(E/f"benedict_q_e26_H_s{s}__ep20000__task3_cc_val550731.csv")['score'] for s in range(20,25)])
Ftile = np.tile(f['score'], 5)
m,lo,hi = mean_ci(allH - Ftile)
print(f"  pooled-round paired diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   (n={len(allH)} pseudo-rounds)")

"""Training-log trajectory of CRATE_DESTROYED, paired per run index.

A training curve is not a result (MEASUREMENT.md) -- this is used only to ask
whether PLB's crate advantage *develops* over training or appears at 20 000
out of nowhere. n = 5 runs x 20 000 episodes per arm.
"""
import csv, numpy as np

BASE = "results/train/task4_tournament/benedict_task3__q_{arm}_s{seed}.csv"
SEEDS = [100, 101, 102, 103, 104]
ARMS = {"ctl": "e33_ctl", "OPP": "e36_OPP", "PLB": "e36_PLB"}
COL = "CRATE_DESTROYED"


def series(arm, seed, col=COL):
    with open(BASE.format(arm=arm, seed=seed)) as fh:
        return np.array([float(r[col]) for r in csv.DictReader(fh)])


W = 2500
for col in (COL, "COIN_COLLECTED", "KILLED_SELF", "SURVIVED_ROUND"):
    print(f"\n=== {col} : per-2500-episode window mean, paired diff vs ctl (n=5) ===")
    ctl = np.array([series(ARMS["ctl"], s, col) for s in SEEDS])
    n = ctl.shape[1]
    hdr = "window        ctl  " + "".join(f"{a:>22}" for a in ("OPP", "PLB"))
    print(hdr)
    for w0 in range(0, n, W):
        sl = slice(w0, w0 + W)
        c = ctl[:, sl].mean(axis=1)
        row = f"{w0:>6}-{min(w0+W,n):<6}{c.mean():7.3f}"
        for a in ("OPP", "PLB"):
            arr = np.array([series(ARMS[a], s, col)[sl].mean() for s in SEEDS])
            d = arr - c
            se = d.std(ddof=1) / np.sqrt(5)
            row += f"  {arr.mean():7.3f} {d.mean():+6.3f}+-{2.776*se:.3f}"
        print(row)

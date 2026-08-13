"""Structure of each checkpoint: value scale, within-row margin, argmax agreement.

Restricted to rows with any nonzero value (the parent's ~2400 live rows plus
whatever training added), and separately to the rows the *greedy* policy of the
frozen rung-2 table actually visits, which are the ones that decide play.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CK = ROOT / "checkpoints" / "benedict_task3"

names = ["_rung2ship",
         "_e25_A_s10__ep20000", "_e25_A_s10__ep40000",
         "_e25_B_s10__ep20000", "_e25_B_s10__ep40000",
         "_e26_S_s20__ep20000", "_e26_S_s20__ep40000",
         "_e26_H_s20__ep20000", "_e26_H_s20__ep40000"]

parent = np.load(CK / "q_table_rung2ship.npy")
plive = np.abs(parent).sum(axis=1) > 0
print(f"parent live rows: {plive.sum()}")

hdr = (f"{'table':<24} {'live':>6} {'Q med':>8} {'Q p5':>8} {'Q p95':>8} "
       f"{'|Q| med':>8} {'margin med':>11} {'margin p90':>11} {'argmax==parent':>15}")
print(hdr)
for n in names:
    q = np.load(CK / f"q_table{n}.npy")
    live = np.abs(q).sum(axis=1) > 0
    sub = q[plive]                      # compare on the same row set
    srt = np.sort(sub, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    agree = (np.argmax(sub, axis=1) == np.argmax(parent[plive], axis=1)).mean()
    print(f"{n:<24} {live.sum():>6} {np.median(sub):8.4f} {np.percentile(sub,5):8.4f} "
          f"{np.percentile(sub,95):8.4f} {np.median(np.abs(sub)):8.4f} "
          f"{np.median(margin):11.6f} {np.percentile(margin,90):11.6f} {agree:15.3f}")

print("\nmargin quantiles on parent-live rows")
for n in names:
    q = np.load(CK / f"q_table{n}.npy")[plive]
    srt = np.sort(q, axis=1)
    m = srt[:, -1] - srt[:, -2]
    qs = np.percentile(m, [10, 25, 50, 75, 90, 99])
    print(f"{n:<24} " + "  ".join(f"{v:9.5f}" for v in qs)
          + f"   frac<1e-3 {np.mean(m < 1e-3):.3f}  frac<1e-2 {np.mean(m < 1e-2):.3f}")

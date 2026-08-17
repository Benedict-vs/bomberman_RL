#!/usr/bin/env python3
"""E38 P3 — does long training erode decision margins?

    uv run python scratchpad/benedict/e38_margins.py

E18 diagnosed 200 000 episodes turning 97.31 into 62.85 as "re-rolling near-tie
rows", and E22 corroborated from the other side: alpha -> 1.0 tripled the
thin-margin fraction and cost the best seed 88 crates. Neither measured the
margin directly across a horizon. This does.

A row counts only if training *changed* it -- comparing against the warm parent,
because an untouched row sits at the parent value and is not an estimate. Danger
rows only (digit 5 > 0), since that is where the E37 feature lives and where the
deaths are.

P3 as pre-registered: the thin-margin share stays below 1.5x its @20 000 value
at every checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CKPT = Path("checkpoints/benedict_task4")
SEEDS = (120, 121, 122, 123, 124)
EPS = (20_000, 40_000, 80_000, 160_000, 300_000)
THIN = 0.01
P3_FACTOR = 1.5


def margins(path: Path, parent: np.ndarray) -> tuple[float, int, float]:
    """(thin-margin share, updated danger rows, mean margin)."""
    q = np.load(path)
    idx = np.arange(q.shape[0])
    danger = ((idx // 50) % 5) > 0
    updated = np.abs(q - parent).sum(1) > 1e-12
    rows = q[danger & updated]
    if not len(rows):
        return float("nan"), 0, float("nan")
    top2 = np.partition(rows, -2, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    return float((gap < THIN).mean()), len(rows), float(gap.mean())


def main() -> None:
    parent = np.load(CKPT / "q_table_parent.npy")
    print(f"thin margin = |best - second| < {THIN}, danger rows training changed\n")
    print(f"{'episodes':>9} {'thin share':>11} {'vs @20k':>9} {'rows':>8} {'mean margin':>12}")

    base = None
    for ep in EPS:
        shares, rows, gaps = [], [], []
        for s in SEEDS:
            p = CKPT / f"q_table_e38_s{s}__ep{ep}.npy"
            if not p.is_file():
                continue
            sh, n, g = margins(p, parent)
            shares.append(sh); rows.append(n); gaps.append(g)
        if not shares:
            print(f"{ep:>9}   (no checkpoints yet)")
            continue
        m = float(np.mean(shares))
        if base is None:
            base = m
        print(f"{ep:>9} {m:>11.4f} {m / base:>8.2f}x {int(np.mean(rows)):>8} {np.mean(gaps):>12.4f}")

    print(f"\nP3 passes while the ratio stays below {P3_FACTOR}x.")
    print("A ratio above it with score holding means the erosion is real but not")
    print("yet expressed -- 300 000 would then be a cliff edge, not a safe horizon.")


if __name__ == "__main__":
    main()

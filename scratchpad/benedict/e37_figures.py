#!/usr/bin/env python3
"""Report figures for rung 4.

    uv run python scratchpad/benedict/e37_figures.py

Writes to `results/figures/` (gitignored -- figures are pure functions of the
committed CSVs). The ship-vs-previous comparison is not here; `tools/analyze.py
--compare ... --plot` already makes it.

Two things these deliberately show that a bar chart would hide: the confidence
intervals, because three of the four arm contrasts include zero, and the
checkpoint trajectory, because "the effect builds while the controls flatten" is
the part that separates a real effect from a lucky checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import e37_analyse as A                                          # noqa: E402

OUT = Path("results/figures")
ARMS = [("PLB2", "(x+y) mod 4\nlattice + stripe"),
        ("PAR", "(x+y) mod 2\nlattice only"),
        ("SHF", "balanced relabel\nno lattice")]
EPS = (5000, 10000, 20000)


def paired(data, arm: str, metric: str = "score"):
    seeds = sorted(set(data[arm]) & set(data["ctl2"]))
    d = np.array([data[arm][s][metric] - data["ctl2"][s][metric] for s in seeds])
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), d.mean() - t * sem, d.mean() + t * sem, int((d > 0).sum()), len(d)


def forest(data) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ys = range(len(ARMS))
    for y, (arm, label) in zip(ys, ARMS):
        m, lo, hi, pos, n = paired(data, arm)
        solid = lo > 0
        ax.plot([lo, hi], [y, y], color="#22303f" if solid else "#9aa7b4", lw=2.4,
                solid_capstyle="butt", zorder=2)
        ax.plot([m], [y], "o", ms=9, color="#c2410c" if solid else "#9aa7b4", zorder=3)
        ax.text(hi + 0.03, y, f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {pos}/{n} seeds",
                va="center", fontsize=8.5, color="#22303f" if solid else "#6b7785")
    ax.axvline(0, color="#22303f", lw=1, zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lab for _, lab in ARMS], fontsize=8.5)
    ax.set_ylim(-0.6, len(ARMS) - 0.4)
    ax.set_xlim(-0.35, 1.05)
    ax.set_xlabel("paired difference in score vs the control  (95 % CI, n = 15 seeds)", fontsize=9)
    ax.set_title("What digit 8 carries in the danger rows", fontsize=11, loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(OUT / "task4_arm_forest.png", dpi=200)
    print("wrote", OUT / "task4_arm_forest.png")


def trajectory() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colours = {"PLB2": "#c2410c", "PAR": "#0369a1", "SHF": "#6b7785"}
    for arm, label in ARMS:
        ms, los, his = [], [], []
        for ep in EPS:
            d = A.load(ep)[0]
            m, lo, hi, _, _ = paired(d, arm)
            ms.append(m); los.append(lo); his.append(hi)
        ax.plot(EPS, ms, "-o", color=colours[arm], lw=2, ms=5,
                label=label.replace("\n", " — "))
        ax.fill_between(EPS, los, his, color=colours[arm], alpha=0.10, lw=0)
    ax.axhline(0, color="#22303f", lw=1)
    ax.set_xticks(EPS)
    ax.set_xticklabels([f"{e // 1000}k" for e in EPS])
    ax.set_xlabel("training episodes", fontsize=9)
    ax.set_ylabel("paired score vs control", fontsize=9)
    ax.set_title("Only the lattice arm keeps building", fontsize=11, loc="left", weight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "task4_checkpoint_trajectory.png", dpi=200)
    print("wrote", OUT / "task4_checkpoint_trajectory.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    forest(A.load(20000)[0])
    trajectory()

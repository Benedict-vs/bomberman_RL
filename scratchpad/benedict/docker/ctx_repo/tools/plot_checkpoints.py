#!/usr/bin/env python3
"""Learning curves from *checkpoint evaluations*, not from the training log.

``trainlog.py`` plots what happened while the agent was learning: eps-greedy,
against a table that changes under it. That curve answers "did it converge",
and AGENTS.md is blunt that it is not a result -- E01 measured 48.2 coins at
the end of training and 1.45 in the evaluation of the same model.

This plots the other curve: each checkpoint evaluated at eps = 0 on the fixed
arena set, which is the same measurement every reported number in
``results/eval/`` comes from. It is what shows whether a run had converged when
it stopped -- the question E20 turned on, where the incumbent had flattened
(+4.51 over the last 60 000 episodes) and the finer map had not (+15.05).

Reads the labels written by ``evaluate.py`` and expects the convention this
project uses:

    <prefix>_<arm>_s<seed>__ep<episodes>__<task>.csv

Each arm gets one bold line (the mean over seeds) plus a shaded band of +-1 sd
and one faint line per seed, because on this project the spread between seeds
has repeatedly been the finding rather than the noise.

    uv run python tools/plot_checkpoints.py --metric crates \
        'results/eval/task2_crates/benedict_q_e20_dist_s*__ep*__task2.csv'
    uv run python tools/plot_checkpoints.py --metric crates \
        --reference results/eval/baselines/ref_rule_based_agent__task2.csv \
        'results/eval/task2_crates/benedict_q_e2*_s*__ep*__task2.csv'

Needs matplotlib (``uv add matplotlib``) -- analysis only, keep it out of the
submitted requirements.txt.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# <arm>_s<seed>__ep<episodes>__<task>
LABEL = re.compile(r"^(?P<prefix>.+?)_(?P<arm>.+)_s(?P<seed>\d+)__ep(?P<ep>\d+)__")


def mean_metric(path: Path, metric: str, agent_slot: int = 0) -> float | None:
    """Mean of `metric` over the rounds of one evaluation CSV.

    Slot 0 is ours by the convention `evaluate.py --agents <ours> ...` sets.
    A file without the column is skipped rather than crashing the whole plot:
    reference CSVs from another rung legitimately lack some metrics.
    """

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or metric not in rows[0]:
        return None
    if "agent_index" in rows[0]:
        rows = [r for r in rows if int(r["agent_index"]) == agent_slot] or rows
    return st.mean(float(r[metric]) for r in rows)


def collect(paths: list[str], metric: str) -> dict[str, dict[int, dict[str, float]]]:
    """arm -> episode -> {seed: value}, from the filename convention."""

    out: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        match = LABEL.match(Path(path).stem)
        if not match:
            continue
        value = mean_metric(Path(path), metric)
        if value is None:
            continue
        out[match["arm"]][int(match["ep"])][match["seed"]] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="evaluation CSVs, globs allowed")
    parser.add_argument("--metric", default="crates")
    parser.add_argument("--reference", action="append", default=[],
                        help="CSV to draw as a horizontal reference line; repeatable")
    parser.add_argument("--out", help="default: results/figures/curve_<metric>.png")
    parser.add_argument("--no-seeds", action="store_true",
                        help="mean and band only, without the per-seed lines")
    parser.add_argument("--table", action="store_true",
                        help="also print the numbers as a Markdown table")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.files:
        paths += sorted(glob.glob(pattern)) or [pattern]
    data = collect(paths, args.metric)
    if not data:
        parser.error(f"no files matched, or none carried a '{args.metric}' column")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for arm in sorted(data):
        episodes = sorted(data[arm])
        seeds = sorted({s for ep in episodes for s in data[arm][ep]})
        means = [st.mean(data[arm][ep].values()) for ep in episodes]
        sds = [st.stdev(data[arm][ep].values()) if len(data[arm][ep]) > 1 else 0.0
               for ep in episodes]

        line, = ax.plot(episodes, means, marker="o", linewidth=2.2, label=arm)
        ax.fill_between(episodes,
                        [m - s for m, s in zip(means, sds)],
                        [m + s for m, s in zip(means, sds)],
                        color=line.get_color(), alpha=0.15, linewidth=0)
        if not args.no_seeds:
            for seed in seeds:
                xs = [ep for ep in episodes if seed in data[arm][ep]]
                ax.plot(xs, [data[arm][ep][seed] for ep in xs],
                        color=line.get_color(), alpha=0.35, linewidth=0.9)

    for ref in args.reference:
        value = mean_metric(Path(ref), args.metric)
        if value is None:
            continue
        ax.axhline(value, linestyle="--", linewidth=1.2, color="0.35")
        ax.annotate(f"{Path(ref).stem.split('__')[0].replace('ref_', '')}: {value:.1f}",
                    xy=(1.0, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color="0.35")

    ax.set_xlabel("training episodes")
    ax.set_ylabel(f"{args.metric} per round (eps = 0, 300 rounds, seed 20260731)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "figures" / f"curve_{args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    if args.table:
        arms = sorted(data)
        all_eps = sorted({ep for arm in arms for ep in data[arm]})
        print("\n| episodes | " + " | ".join(arms) + " |")
        print("|---" * (len(arms) + 1) + "|")
        for ep in all_eps:
            cells = []
            for arm in arms:
                vals = data[arm].get(ep)
                cells.append(f"{st.mean(vals.values()):.2f} ± "
                             f"{st.stdev(vals.values()):.2f}" if vals and len(vals) > 1
                             else (f"{st.mean(vals.values()):.2f}" if vals else "—"))
            print(f"| {ep} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

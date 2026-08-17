#!/usr/bin/env python3
"""E37 — where is the sweep right now, and when does it land.

    uv run python scratchpad/benedict/e37_status.py

Reads the training CSVs (one row per episode) and the evaluation CSVs, and
extrapolates a finish time from the *observed* episode rate rather than from a
prior sweep's. E36's 1.85 h/batch was a bad baseline for E37: its OPP arm pays
an extra BFS on every danger step, where all four E37 arms are O(1) arithmetic
on digit 8, so E37 runs materially faster.

The rate is not constant -- early episodes are short because the agent dies
quickly and epsilon is still high -- so the estimate is deliberately based on
the most recent slice of each run, not on its lifetime average.
"""

from __future__ import annotations

import csv
import glob
import os
import time
from collections import defaultdict

EPISODES = 20_000
BATCH = 10
TOTAL_RUNS = 60
TRAIN_GLOB = "results/train/task4_tournament/benedict_task3__q_e37_*.csv"
EVAL_GLOB = "results/eval/task4_tournament/benedict_q_e37_*.csv"
OUT_GLOB = "results/train/task4_tournament/e37_*.out"


def tail_rate(rows: list[dict], frac: float = 0.25) -> float:
    """Episodes per second over the last `frac` of the run so far.

    Late episodes are longer than early ones, so the lifetime mean flatters the
    remaining time. Measuring the recent slice is the honest extrapolation.
    """
    if len(rows) < 20:
        return 0.0
    cut = max(1, int(len(rows) * (1 - frac)))
    d_ep = len(rows) - cut
    d_t = float(rows[-1]["wall_clock_s"]) - float(rows[cut]["wall_clock_s"])
    return d_ep / d_t if d_t > 0 else 0.0


def main() -> None:
    runs: dict[str, tuple[int, float, float]] = {}
    for path in sorted(glob.glob(TRAIN_GLOB)):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        name = os.path.basename(path).split("__")[-1].replace(".csv", "")
        runs[name] = (len(rows), float(rows[-1]["wall_clock_s"]), tail_rate(rows))

    by_arm: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for name, v in runs.items():
        by_arm[name.split("_")[2]].append(v)

    print(f"\nE37 training -- {len(runs)}/{TOTAL_RUNS} runs started, "
          f"{time.strftime('%H:%M:%S')}\n")
    print(f"{'arm':>6} {'runs':>5} {'mean ep':>9} {'done':>7} {'ep/s':>7} {'elapsed':>9}")
    for arm in ("ctl2", "PLB2", "PAR", "SHF"):
        if arm not in by_arm:
            continue
        v = by_arm[arm]
        mean_ep = sum(x[0] for x in v) / len(v)
        mean_wall = sum(x[1] for x in v) / len(v)
        rate = sum(x[2] for x in v) / len(v)
        print(f"{arm:>6} {len(v):>5} {mean_ep:>9.0f} "
              f"{mean_ep / EPISODES:>6.1%} {rate:>7.2f} {mean_wall / 3600:>8.2f} h")

    # Extrapolate: finish the in-flight batch, then the batches not yet started.
    live = [v for v in runs.values() if v[0] < EPISODES]
    if live:
        rate = sum(v[2] for v in live) / len(live)
        if rate > 0:
            left_now = max(EPISODES - min(v[0] for v in live), 0) / rate
            batches_left = (TOTAL_RUNS - len(runs)) / BATCH
            per_batch = EPISODES / rate
            eta = left_now + batches_left * per_batch
            print(f"\n  in-flight batch finishes in {left_now / 3600:.2f} h "
                  f"({time.strftime('%H:%M', time.localtime(time.time() + left_now))})")
            print(f"  {batches_left:.0f} batches not started x {per_batch / 3600:.2f} h")
            print(f"  SWEEP ETA {eta / 3600:.2f} h  -> "
                  f"{time.strftime('%a %H:%M', time.localtime(time.time() + eta))}")
    else:
        print("\n  no run below 20 000 episodes -- training looks finished")

    n_eval = len(glob.glob(EVAL_GLOB))
    print(f"\nE37 evaluations on disk: {n_eval}/180 "
          f"({'primary @20000 complete' if n_eval >= 60 else 'primary needs 60'})")

    bad = [p for p in glob.glob(OUT_GLOB) if "Traceback" in open(p, errors="ignore").read()]
    print(f"tracebacks: {', '.join(os.path.basename(p) for p in bad) if bad else 'none'}\n")


if __name__ == "__main__":
    main()

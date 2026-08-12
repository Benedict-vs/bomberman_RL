"""E25 arm-level comparison. The unit of analysis is the training run, not the round.

`tools/analyze.py --compare` bootstraps over rounds, which is right when two tables
play the same arenas and wrong here: the thing that varies between arms is the
*training run*, and five runs is the sample size. Pooling 300 rounds x 5 runs as
1500 independent observations would understate the CI by roughly sqrt(300).

Arms A and B share BM_RUN_INDEX seed for seed, so B - A is paired over five pairs
and gets a paired t interval. n = 5 is small on purpose -- the interval is wide and
should be read as such.

Run:  uv run python scratchpad/benedict/e25_arms.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "eval" / "task3_opponents"

AGENT = "benedict_task3"
SEEDS = (10, 11, 12, 13, 14)
METRICS = ("score", "coins", "kills", "suicides", "killed_by_opponent",
           "survived", "crates", "invalid", "steps")
# Two-sided 95 % t quantiles for n - 1 degrees of freedom.
T95 = {4: 2.776, 3: 3.182, 2: 4.303}


def run_mean(path: Path) -> dict[str, float] | None:
    """Mean of each metric over one evaluation's rounds, for our agent only."""
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r["code"] == AGENT]
    if not rows:
        return None
    return {m: st.mean(float(r[m]) for r in rows) for m in METRICS}


def interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95 % t interval half-width over runs."""
    n = len(values)
    mean = st.mean(values)
    if n < 2:
        return mean, mean, mean
    half = T95.get(n - 1, 2.776) * st.stdev(values) / n ** 0.5
    return mean, mean - half, mean + half


def label(arm: str, seed: int, ep: int, field: str) -> Path:
    return EVAL / f"benedict_q_e25_{arm}_s{seed}__ep{ep}__task3_{field}_val550731.csv"


def report(field: str, ep: int, floor: dict[str, float]) -> None:
    print(f"\n{'=' * 78}\n{field}  @ episode {ep}   (n = {len(SEEDS)} training runs per arm)\n{'=' * 78}")
    per_arm: dict[str, dict[str, list[float]]] = {}
    for arm in ("A", "B"):
        runs = [run_mean(label(arm, s, ep, field)) for s in SEEDS]
        missing = [s for s, r in zip(SEEDS, runs) if r is None]
        if missing:
            print(f"  arm {arm}: MISSING seeds {missing}")
        runs = [r for r in runs if r is not None]
        if not runs:
            return
        per_arm[arm] = {m: [r[m] for r in runs] for m in METRICS}

    head = f"  {'metric':<20}{'floor':>9}{'arm A (95% CI)':>26}{'arm B (95% CI)':>26}"
    print(head)
    for m in METRICS:
        cells = []
        for arm in ("A", "B"):
            mean, lo, hi = interval(per_arm[arm][m])
            cells.append(f"{mean:8.3f} [{lo:7.3f},{hi:7.3f}]")
        f = floor.get(m)
        fcell = f"{f:9.3f}" if f is not None else " " * 9
        print(f"  {m:<20}{fcell}{cells[0]:>26}{cells[1]:>26}")

    print(f"\n  paired B - A over the {len(SEEDS)} shared training seeds:")
    for m in METRICS:
        diffs = [b - a for a, b in zip(per_arm["A"][m], per_arm["B"][m])]
        mean, lo, hi = interval(diffs)
        verdict = "excludes 0" if lo > 0 or hi < 0 else "not demonstrated"
        print(f"  {m:<20}{mean:+9.3f}  [{lo:+8.3f},{hi:+8.3f}]   {verdict}")


# The shipped rung-2 table on the *same* held-out seed the arms are measured on.
# E24's numbers are on the dev seed 20260731 and would have made every comparison
# below confounded by the world seed; these were re-measured at 550731 for that
# reason. They land within 0.06 of E24 on every metric, which is a useful check
# that the floor is a property of the agent and not of the seed.
FLOOR_CC = {"score": 2.517, "coins": 2.183, "kills": 0.067, "suicides": 0.220,
            "killed_by_opponent": 0.343, "survived": 0.437, "crates": 27.34,
            "invalid": 25.29, "steps": 229.0}
FLOOR_RB = {"score": 2.663, "coins": 2.297, "kills": 0.073, "suicides": 0.360,
            "killed_by_opponent": 0.510, "survived": 0.130, "crates": 27.83,
            "invalid": 8.38, "steps": 156.0}

if __name__ == "__main__":
    report("cc", 20000, FLOOR_CC)
    report("cc", 40000, FLOOR_CC)
    report("rb", 40000, FLOOR_RB)
    print("\nfloor = E24, shipped rung-2 table on the *dev* seed 20260731; these arms are on "
          "the held-out 550731, so the floor is context and is never differenced against.")

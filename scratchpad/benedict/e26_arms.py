"""E26 arm-level comparison. Unit of analysis is the training run (n = 5), not the round.

Arm F is a single frozen table with no training, so it has no run-to-run spread and
gets a per-round bootstrap CI instead -- the two kinds of interval are NOT
interchangeable and are labelled as such. F is a fixed reference, so comparisons
against it are unpaired; S vs H are paired by BM_RUN_INDEX.

Run:  uv run python scratchpad/benedict/e26_arms.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "eval" / "task3_opponents"
AGENT = "benedict_task3"
SEEDS = (20, 21, 22, 23, 24)
METRICS = ("score", "coins", "kills", "suicides", "killed_by_opponent",
           "survived", "crates", "invalid", "steps", "won")
T95 = {4: 2.776, 3: 3.182, 2: 4.303}

# Floors, both on seed 550731, 300 rounds (E25 section of experiments/benedict.md).
FLOOR = {"cc": 2.517, "rb": 2.663}
REFERENCE = {"cc": 2.753, "rb": 3.290}   # rule_based_agent in our slot, E24 (dev seed)


def rows(path: Path):
    if not path.exists():
        return None
    r = [x for x in csv.DictReader(path.open()) if x["code"] == AGENT]
    return r or None


def run_mean(path: Path):
    r = rows(path)
    return None if r is None else {m: st.mean(float(x[m]) for x in r) for m in METRICS}


def t_interval(vals: list[float]):
    n = len(vals)
    mean = st.mean(vals)
    if n < 2:
        return mean, float("nan"), float("nan")
    h = T95.get(n - 1, 2.776) * st.stdev(vals) / n ** 0.5
    return mean, mean - h, mean + h


def bootstrap(vals: list[float], iters: int = 2000):
    """Percentile bootstrap over rounds, for the single frozen table (arm F)."""
    import random
    rnd = random.Random(20260731)
    n = len(vals)
    means = sorted(st.mean(rnd.choices(vals, k=n)) for _ in range(iters))
    return st.mean(vals), means[int(0.025 * iters)], means[int(0.975 * iters)]


def show(title: str, cells: dict[str, tuple[float, float, float]]):
    print(f"\n  {title}")
    for name, (m, lo, hi) in cells.items():
        ci = f"[{lo:7.3f},{hi:7.3f}]" if lo == lo else "  (single run)"
        print(f"    {name:<28}{m:8.3f} {ci}")


def report(field: str, ep: int):
    print(f"\n{'=' * 84}\n{field}  @ episode {ep}   (n = 5 runs per trained arm)\n{'=' * 84}")
    print(f"  floor (frozen rung-2, HUNT off) = {FLOOR[field]:.3f}   "
          f"rule_based_agent in our slot = {REFERENCE[field]:.3f}")

    per_arm = {}
    for arm in ("S", "H"):
        runs = [run_mean(EVAL / f"benedict_q_e26_{arm}_s{s}__ep{ep}__task3_{field}_val550731.csv")
                for s in SEEDS]
        if any(r is None for r in runs):
            print(f"  arm {arm}: MISSING "
                  f"{[s for s, r in zip(SEEDS, runs) if r is None]}")
        runs = [r for r in runs if r is not None]
        if runs:
            per_arm[arm] = {m: [r[m] for r in runs] for m in METRICS}

    f = rows(EVAL / f"benedict_q_e26_F_frozen__task3_{field}_val550731.csv")

    print(f"\n  {'metric':<20}{'arm F frozen':>26}{'arm S (95% CI)':>26}{'arm H (95% CI)':>26}")
    for m in METRICS:
        line = f"  {m:<20}"
        if f:
            fm, flo, fhi = bootstrap([float(x[m]) for x in f])
            line += f"{fm:8.3f} [{flo:6.2f},{fhi:6.2f}]".rjust(26)
        else:
            line += " " * 26
        for arm in ("S", "H"):
            if arm in per_arm:
                mm, lo, hi = t_interval(per_arm[arm][m])
                line += f"{mm:8.3f} [{lo:6.2f},{hi:6.2f}]".rjust(26)
            else:
                line += " " * 26
        print(line)

    if "S" in per_arm and "H" in per_arm:
        print(f"\n  paired H - S over the {len(SEEDS)} shared training seeds:")
        for m in METRICS:
            d = [h - s for s, h in zip(per_arm["S"][m], per_arm["H"][m])]
            mm, lo, hi = t_interval(d)
            print(f"  {m:<20}{mm:+9.3f}  [{lo:+8.3f},{hi:+8.3f}]   "
                  f"{'excludes 0' if lo > 0 or hi < 0 else 'not demonstrated'}")

    # P1: does arm H clear the floor? Unpaired -- the floor is a fixed number.
    if "H" in per_arm:
        mm, lo, hi = t_interval([v - FLOOR[field] for v in per_arm["H"]["score"]])
        print(f"\n  P1  arm H score - floor ({FLOOR[field]:.3f}): {mm:+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"   -> {'CLEARS' if lo > 0 else 'not demonstrated'}")


if __name__ == "__main__":
    report("cc", 20000)
    report("cc", 40000)
    report("rb", 20000)

    print(f"\n{'=' * 84}\narm F and its control, single frozen tables, per-round bootstrap\n{'=' * 84}")
    for lbl, name in (("F, HUNT on,  cc", "F_frozen__task3_cc_val550731"),
                      ("F, HUNT off, cc", "F_control_huntoff__task3_cc_val550731"),
                      ("F, HUNT on,  rule_based", "F_frozen__task3_rb_val550731"),
                      ("F, HUNT on,  peaceful", "F_frozen__task3_peaceful_val550731")):
        r = rows(EVAL / f"benedict_q_e26_{name}.csv")
        if not r:
            print(f"  {lbl:<26} MISSING")
            continue
        out = []
        for m in ("score", "kills", "invalid", "crates", "survived", "won"):
            mm, lo, hi = bootstrap([float(x[m]) for x in r])
            out.append(f"{m} {mm:.3f} [{lo:.2f},{hi:.2f}]")
        print(f"  {lbl:<26} " + "  ".join(out))

"""E27 arm comparison. Unit of analysis is the training run (n = 5), not the round.

Baselines are arm F -- the frozen rung-2 table read through the rung-3 feature map,
E26 -- measured on the same validation seed and round count. F is a single fixed
table, so comparisons against it are unpaired; the arms share BM_RUN_INDEX and are
paired against each other.
"""
from __future__ import annotations
import csv, statistics as st
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "results/eval/task3_opponents"
AGENT, SEEDS = "benedict_task3", (50, 51, 52, 53, 54)
METRICS = ("score", "coins", "kills", "suicides", "survived", "crates", "invalid", "steps", "won")
T95 = {4: 2.776, 3: 3.182, 2: 4.303}


def rows(p: Path):
    if not p.exists():
        return None
    r = [x for x in csv.DictReader(p.open()) if x["code"] == AGENT]
    return r or None


def run_mean(p: Path):
    r = rows(p)
    return None if r is None else {m: st.mean(float(x[m]) for x in r) for m in METRICS}


def ci(v):
    n = len(v)
    m = st.mean(v)
    if n < 2:
        return m, float("nan"), float("nan")
    h = T95.get(n - 1, 2.776) * st.stdev(v) / n ** 0.5
    return m, m - h, m + h


def baseline(field):
    r = rows(EVAL / f"benedict_q_e26_F_frozen__task3_{field}_val550731.csv")
    return {m: st.mean(float(x[m]) for x in r) for m in METRICS} if r else None


def report(field, ep):
    F = baseline(field)
    print(f"\n{'='*92}\n{field} @ ep{ep}   n=5 runs/arm   arm F (frozen ship) score={F['score']:.3f}\n{'='*92}")
    data = {}
    for arm in ("C03", "C10", "S03"):
        runs = [run_mean(EVAL / f"benedict_q_e27_{arm}_s{s}__ep{ep}__task3_{field}_val550731.csv") for s in SEEDS]
        miss = [s for s, r in zip(SEEDS, runs) if r is None]
        if miss:
            print(f"  {arm}: MISSING {miss}")
        runs = [r for r in runs if r]
        if runs:
            data[arm] = {m: [r[m] for r in runs] for m in METRICS}
    print(f"  {'metric':<12}{'arm F':>10}" + "".join(f"{a+' (95% CI)':>26}" for a in data))
    for m in METRICS:
        line = f"  {m:<12}{F[m]:>10.3f}"
        for a in data:
            mm, lo, hi = ci(data[a][m])
            line += f"{mm:8.3f} [{lo:6.2f},{hi:6.2f}]".rjust(26)
        print(line)
    for a in data:
        mm, lo, hi = ci([v - F["score"] for v in data[a]["score"]])
        verdict = "BEATS F" if lo > 0 else ("WORSE" if hi < 0 else "not demonstrated")
        print(f"  {a} score - F: {mm:+.3f} [{lo:+.3f},{hi:+.3f}]  -> {verdict}")
    if "C10" in data and "C03" in data:
        for m in ("score", "crates"):
            mm, lo, hi = ci([c - b for b, c in zip(data["C03"][m], data["C10"][m])])
            print(f"  paired C10-C03 {m}: {mm:+.3f} [{lo:+.3f},{hi:+.3f}]")
    if "S03" in data and "C03" in data:
        mm, lo, hi = ci([s - b for b, s in zip(data["C03"]["score"], data["S03"]["score"])])
        print(f"  paired S03-C03 score: {mm:+.3f} [{lo:+.3f},{hi:+.3f}]   (P4)")
    return data


if __name__ == "__main__":
    d20 = report("cc", 20000)
    d40 = report("cc", 40000)
    report("rb", 20000)
    print(f"\n{'='*92}\nP5 horizon guard: score retained from ep20000 to ep40000 (cc)\n{'='*92}")
    for a in d20:
        if a in d40:
            keep = [b / x if x else float('nan') for x, b in zip(d20[a]["score"], d40[a]["score"])]
            m, lo, hi = ci(keep)
            print(f"  {a}: {100*m:5.1f} % retained [{100*lo:.1f},{100*hi:.1f}]  "
                  f"-> {'PASS' if m >= 0.75 else 'FAIL (>25% lost)'}")

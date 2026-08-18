#!/usr/bin/env python3
"""E41 scoring, against the predictions as pre-registered in experiments/benedict.md.

Primary is the paired within-round margin, not our absolute score: our score swings
3.95 -> 14.49 purely on who else is on the board, and the crate pool is zero-sum, so
"our score went up" is near-unfalsifiable. Every difference carries a t and a
sign-flip p (E40 / audit 10: analyze.py's fixed bootstrap seed makes a boundary
verdict look deterministic when it is a coin flip).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"
SHIPPED_VS_RB = 3.949          # experiments/benedict_task4.md section 1
EXT = ["ext_xiaoxiae_binary_v6", "ext_xiaoxiae_bindist_v2",
       "ext_aielka_ql_atom", "ext_lijesse_featureeverything"]


def rounds(path: Path):
    by = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            by[int(r["round"])].append(r)
    return by


def agg(rows, code, key, how=sum):
    vals = [float(r[key]) for r in rows if r["code"] == code]
    return how(vals) if vals else 0.0


def ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    rng = np.random.default_rng(12345)
    b = x[rng.integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    null = (np.random.default_rng(0).choice([-1.0, 1.0], size=(5000, len(x))) * x).mean(axis=1)
    p = float((np.abs(null) >= abs(m)).mean())
    return m, lo, hi, m / se if se else float("nan"), p


def show(name, x, bar=None):
    m, lo, hi, t, p = ci(x)
    ok = lo * hi > 0 and p < 0.05
    verdict = ("BESSER" if m > 0 else "SCHLECHTER") if ok else "nicht gezeigt"
    extra = ""
    if bar is not None:
        extra = f"  | bar {bar:+.2f}: {'CLEARS' if (m > bar and ok) else 'FAILS'}"
    print(f"  {name:<24}{m:+8.3f} [{lo:+.3f}, {hi:+.3f}]  t={t:+6.2f} p={p:.4f}  {verdict}{extra}")


print("=" * 96)
print("E41 -- the shipped table against four third-party agents.  n=1000, ship seed 990731")
print("=" * 96)

print("\n## P2 -- the calibration constant: THEIR agent in OUR slot vs 3x rule_based")
print(f"   (our shipped table in that slot scores {SHIPPED_VS_RB})\n")
print(f"  {'agent':<34}{'score':>8}{'vs ours':>10}{'coins':>8}{'kills':>8}"
      f"{'suic':>7}{'crates':>8}{'inval med':>11}{'thinkmax':>9}{'over%':>6}")
for e in EXT:
    f = EVAL / f"{e.replace('ext_','ext_')}_calib__task4_rb_ship990731.csv"
    if not f.exists():
        print(f"  {e:<34}  (pending)"); continue
    R = rounds(f)
    sc = np.mean([agg(v, e, "score") for v in R.values()])
    inv = np.median([agg(v, e, "invalid") for v in R.values()])
    # `time` is the round's CUMULATIVE think time; evaluate.py:246-247 records the
    # per-step statistics separately, and those are the ones the 0.5 s guard is about.
    tm = max(agg(v, e, "think_max_ms", max) for v in R.values())
    over = sum(int(agg(v, e, "think_over_limit")) for v in R.values())
    steps = sum(agg(v, e, "steps") for v in R.values())
    over = 100.0 * over / max(steps, 1)          # percent of steps, as pre-registered
    print(f"  {e:<34}{sc:8.3f}{sc - SHIPPED_VS_RB:+10.3f}"
          f"{np.mean([agg(v,e,'coins') for v in R.values()]):8.3f}"
          f"{np.mean([agg(v,e,'kills') for v in R.values()]):8.3f}"
          f"{np.mean([agg(v,e,'suicides') for v in R.values()]):7.3f}"
          f"{np.mean([agg(v,e,'crates') for v in R.values()]):8.2f}"
          f"{inv:11.1f}{tm:9.1f}{over:5.2f}%")

print("\n## P1 -- head-to-head: ours + 3x external.  PRIMARY = margin_mean")
for e in EXT:
    f = EVAL / f"benedict_task4_shipped_e37__task4_{e}_ship990731.csv"
    if not f.exists():
        print(f"\n  {e}: (pending)"); continue
    R = rounds(f)
    mine = np.array([agg(v, OURS, "score") for v in R.values()])
    omean = np.array([np.mean([float(r["score"]) for r in v if r["code"] != OURS])
                      for v in R.values()])
    obest = np.array([max(float(r["score"]) for r in v if r["code"] != OURS)
                      for v in R.values()])
    print(f"\n  {e}   (our score {mine.mean():.3f}, their mean {omean.mean():.3f})")
    show("margin_mean", mine - omean)
    show("margin_best", mine - obest)
    # P3: crates per bomb, guard band 1.16 +/- 0.15
    cr = np.mean([agg(v, OURS, "crates") for v in R.values()])
    bo = np.mean([agg(v, OURS, "bombs") for v in R.values()])
    cpb = cr / bo if bo else float("nan")
    flag = "IN BAND" if abs(cpb - 1.16) <= 0.15 else "*** OUT OF BAND -- P3 REFUTED ***"
    print(f"    P3 crates/bomb {cpb:.3f}  (guard 1.16 +/- 0.15)  {flag}")
    # P4: takeable pool
    deaths = np.mean([sum(1 - float(r["survived"]) for r in v if r["code"] != OURS)
                      for v in R.values()])
    suic = np.mean([sum(float(r["suicides"]) for r in v if r["code"] != OURS)
                    for v in R.values()])
    pool = deaths - suic
    print(f"    P4 opp deaths {deaths:.3f} - opp suicides {suic:.3f} = takeable {pool:.3f}"
          f"   (rule_based 0.333; reopen bar 0.5)  {'*** FIRES ***' if pool > 0.5 else 'no'}")
    ourk = np.mean([agg(v, OURS, "kills") for v in R.values()])
    print(f"       our kills {ourk:.3f}, share of takeable {ourk / pool if pool else 0:.1%}")

print("\n## The symmetric bar -- 4x external, no us. Prices the field itself.")
print(f"  (4x rule_based gives won 0.282; our shipped table reaches won 0.406)\n")
for e in EXT:
    f = EVAL / f"ref_{e}__task4_field_{e}_ship990731.csv"
    if not f.exists():
        print(f"  {e:<34}  (pending)"); continue
    R = rounds(f)
    sc = np.mean([float(r["score"]) for v in R.values() for r in v])
    best = np.mean([max(float(r["score"]) for r in v) for v in R.values()])
    surv = np.mean([float(r["survived"]) for v in R.values() for r in v])
    print(f"  {e:<34} mean score {sc:6.3f}   best-of-four {best:6.3f}   survival {surv:.3f}")

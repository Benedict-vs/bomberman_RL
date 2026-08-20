#!/usr/bin/env python3
"""E42 scoring, strictly against the predictions pre-registered in experiments/benedict.md.

Unit of inference is the SEED, not the round: the pre-registered MDE of 0.35 comes from the
between-seed score SD of 0.249 at n = 8, so an arena-paired test over 300 rounds would be a
different (and much more optimistic) claim than the one registered. Arena-paired arm means are
reported alongside as a secondary, clearly labelled.
"""
from __future__ import annotations
import csv, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"
MIX = [200, 201, 202, 203, 204, 205, 206, 207]
CTL = [100, 101, 102, 103, 104, 105, 106, 107]
FIELDS = {"heldout": "HELD OUT: 3x bindist_v2 (never seen in training)",
          "indist":  "in-distribution: the training field",
          "guard":   "regression guard: 3x rule_based"}


def rounds(arm, seed, field):
    f = EVAL / f"e42_{arm}_s{seed}__task4_{field}_val550731.csv"
    if not f.exists():
        return None
    by = defaultdict(list)
    for r in csv.DictReader(open(f)):
        by[int(r["round"])].append(r)
    return by


def metric(by, name):
    """Per-round series for one metric."""
    out = {}
    for rd, rows in by.items():
        me = next(r for r in rows if r["code"] == OURS)
        opp = [float(r["score"]) for r in rows if r["code"] != OURS]
        if name == "margin_mean":  out[rd] = float(me["score"]) - float(np.mean(opp))
        elif name == "margin_best": out[rd] = float(me["score"]) - max(opp)
        elif name == "crates_per_bomb":
            out[rd] = float(me["crates"]) / max(float(me["bombs"]), 1e-9)
        else: out[rd] = float(me[name])
    return out


def perm_p(a, b, n=20000, seed=0):
    """Two-sample permutation test on the mean difference -- the seeds are not paired."""
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(b) - np.mean(a))
    pool = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean() - pool[:na].mean()) >= obs: cnt += 1
    return cnt / n


def boot_ci(a, b, seed=12345, n=10000):
    rng = np.random.default_rng(seed)
    d = [np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def seed_means(arm, seeds, field, name):
    out = []
    for s in seeds:
        by = rounds(arm, s, field)
        if by is None: continue
        out.append(float(np.mean(list(metric(by, name).values()))))
    return np.array(out)


print("=" * 100)
print("E42 -- training against a field that hunts back.  8 seeds/arm, 300 rounds, val seed 550731")
print("=" * 100)

for field, desc in FIELDS.items():
    print(f"\n### {desc}\n")
    print(f"  {'metric':<18}{'control':>10}{'mixed':>10}{'difference':>12}"
          f"{'95% CI':>22}{'perm p':>9}   verdict")
    print("  " + "-" * 92)
    # (metric, higher_is_better) -- AGENTS.md: analyze.py assumes higher is better and prints
    # WORSE for a falling `steps`; the same trap caught this script's first run, which called
    # rising suicides BESSER.
    for name, hib in [("score", True), ("margin_mean", True), ("margin_best", True),
                      ("coins", True), ("kills", True), ("suicides", False),
                      ("killed_by_opponent", False), ("survived", True),
                      ("crates_per_bomb", True)]:
        a = seed_means("ctl", CTL, field, name)
        b = seed_means("mix", MIX, field, name)
        if len(a) == 0 or len(b) == 0:
            print(f"  {name:<18}  (missing)"); continue
        d = b.mean() - a.mean()
        lo, hi = boot_ci(a, b)
        p = perm_p(a, b)
        # AGENTS.md (since c40a902): a row counts only if the CI excludes 0 AND is not
        # fragile. This script did not implement that rule when it was first run, and E42's
        # declared headline (in-dist score -0.161, p = 0.064) passed on the CI alone.
        ci_sig = lo * hi > 0
        fragile = ci_sig != (p < 0.05)
        improved = (d > 0) == hib
        v = ("BESSER" if improved else "SCHLECHTER") if (ci_sig and not fragile) else "nicht gezeigt"
        if fragile:
            v += " (fragile)"
        print(f"  {name:<18}{a.mean():10.3f}{b.mean():10.3f}{d:+12.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:9.4f}   {v}")

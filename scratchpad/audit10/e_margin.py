#!/usr/bin/env python3
"""Audit 10 -- the ceiling was scored on OUR mean score only.

The tournament criterion (AGENTS.md, corrected 2026-08-17) is total score, which is a
*comparison across the four agents in the field*. Killing an opponent raises our score by 5
and also removes that opponent's future earnings. The ceiling CSVs contain all four slots,
so the competitive margin can be read straight off them.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

CEIL = Path("/Users/benedictvonschubert/Projects/bomberman_RL/scratchpad/strategy/ceil")
rng = np.random.default_rng(2024)


def per_round(path):
    rows = list(csv.DictReader(open(path)))
    rounds = {}
    for r in rows:
        rounds.setdefault(int(r["round"]), []).append(r)
    out = {}
    ks = sorted(rounds)
    def arr(f):
        return np.array([f(rounds[k]) for k in ks], dtype=float)
    me = lambda rs: [r for r in rs if r["code"] == "user_agent"][0]
    op = lambda rs: [r for r in rs if r["code"] != "user_agent"]
    out["mine"] = arr(lambda rs: float(me(rs)["score"]))
    out["opp_sum"] = arr(lambda rs: sum(float(r["score"]) for r in op(rs)))
    out["opp_mean"] = out["opp_sum"] / 3.0
    out["opp_best"] = arr(lambda rs: max(float(r["score"]) for r in op(rs)))
    out["opp_kills"] = arr(lambda rs: sum(float(r["kills"]) for r in op(rs)))
    out["opp_coins"] = arr(lambda rs: sum(float(r["coins"]) for r in op(rs)))
    out["opp_suicides"] = arr(lambda rs: sum(float(r["suicides"]) for r in op(rs)))
    out["opp_deaths"] = arr(lambda rs: sum(float(r["died"]) for r in op(rs)))
    out["margin_mean"] = out["mine"] - out["opp_mean"]
    out["margin_best"] = out["mine"] - out["opp_best"]
    out["my_kills"] = arr(lambda rs: float(me(rs)["kills"]))
    return out


def boot(a, b, n=40000):
    d = np.asarray(b) - np.asarray(a)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), d.mean() / se


arms = {k: per_round(CEIL / f"huntceil4k_k{k}__task4_rb_ship990731.csv") for k in (-1, 0, 4)}
keys = ["mine", "opp_mean", "opp_best", "opp_sum", "margin_mean", "margin_best",
        "my_kills", "opp_kills", "opp_coins", "opp_suicides", "opp_deaths"]
print("arm means (n=4000)")
print(f"{'':10s}" + "".join(f"{k:>13s}" for k in keys))
for k, d in arms.items():
    print(f"k={k:<8d}" + "".join(f"{d[m].mean():13.4f}" for m in keys))

for k in (0, 4):
    print(f"\n--- paired k={k} vs k=-1")
    for m in keys:
        mean, lo, hi, t = boot(arms[-1][m], arms[k][m])
        star = "  <-- CI excludes 0" if lo * hi > 0 else ""
        print(f"  {m:12s} d={mean:+8.4f} [{lo:+.4f},{hi:+.4f}]  t={t:+.2f}{star}")

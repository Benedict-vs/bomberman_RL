#!/usr/bin/env python3
"""Audit 10 -- robustness of the margin result, the k=8 'bounds chasing' claim,
and the oracle's conversion rate."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

CEIL = Path("/Users/benedictvonschubert/Projects/bomberman_RL/scratchpad/strategy/ceil")


def per_round(path):
    rows = list(csv.DictReader(open(path)))
    rounds = {}
    for r in rows:
        rounds.setdefault(int(r["round"]), []).append(r)
    ks = sorted(rounds)
    me = lambda rs: [r for r in rs if r["code"] == "user_agent"][0]
    op = lambda rs: [r for r in rs if r["code"] != "user_agent"]
    d = {}
    d["mine"] = np.array([float(me(rounds[k])["score"]) for k in ks])
    d["my_kills"] = np.array([float(me(rounds[k])["kills"]) for k in ks])
    d["opp_mean"] = np.array([np.mean([float(r["score"]) for r in op(rounds[k])]) for k in ks])
    d["opp_best"] = np.array([max(float(r["score"]) for r in op(rounds[k])) for k in ks])
    d["opp_deaths"] = np.array([sum(float(r["died"]) for r in op(rounds[k])) for k in ks])
    d["margin_mean"] = d["mine"] - d["opp_mean"]
    d["margin_best"] = d["mine"] - d["opp_best"]
    return d


def seedscan(a, b, label, n_seeds=200):
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    los = []
    for s in range(n_seeds):
        r = np.random.default_rng(1000 + s)
        m = d[r.integers(0, n, size=(10000, n))].mean(axis=1)
        los.append(np.percentile(m, 2.5))
    los = np.array(los)
    se = d.std(ddof=1) / np.sqrt(n)
    # sign-flip p
    r = np.random.default_rng(11)
    null = (r.choice([-1.0, 1.0], size=(100000, n)) * d).mean(axis=1)
    p = (np.abs(null) >= abs(d.mean())).mean()
    print(f"{label:28s} d={d.mean():+.4f}  t={d.mean()/se:+.2f}  sign-flip p={p:.4f}  "
          f"CI-excludes-0 in {(los>0).mean() if d.mean()>0 else (np.array([0.0])).mean():.0%} "
          f"of 200 bootstrap seeds")


big = {k: per_round(CEIL / f"huntceil4k_k{k}__task4_rb_ship990731.csv") for k in (-1, 0, 4)}
pil = {k: per_round(CEIL / f"huntceil_k{k}__task4_rb_ship990731.csv") for k in (-1, 0, 2, 4, 8)}

print("### robustness of each k=4 vs k=-1 contrast at n=4000")
for m in ("mine", "my_kills", "margin_mean", "margin_best"):
    seedscan(big[-1][m], big[4][m], m)

print("\n### the pilot's 'k=8 bounds chasing harder' claim (n=1000)")
for m in ("mine", "margin_mean", "margin_best", "my_kills"):
    d = pil[8][m] - pil[4][m]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  k8 - k4  {m:12s} {d.mean():+.4f}  t={d.mean()/se:+.2f}")
    d = pil[8][m] - pil[-1][m]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  k8 - kctl {m:12s} {d.mean():+.4f}  t={d.mean()/se:+.2f}")

print("\n### margin on the pilot arms (n=1000) -- monotone in k?")
for k in (-1, 0, 2, 4, 8):
    print(f"  k={k:3d}  mine {pil[k]['mine'].mean():.3f}  margin_mean "
          f"{pil[k]['margin_mean'].mean():+.3f}  margin_best {pil[k]['margin_best'].mean():+.3f}")

print("\n### oracle conversion rate at k=4, n=4000")
hb, hs = 1622, 15733
extra_kills = (big[4]["my_kills"] - big[-1]["my_kills"]).sum()
print(f"  oracle 'inescapable trap' bombs placed: {hb}")
print(f"  extra kills our agent got vs control:   {extra_kills:.0f}  "
      f"({extra_kills/hb:.1%} of the oracle bombs)")
dd = (big[4]["opp_deaths"] - big[-1]["opp_deaths"])
se = dd.std(ddof=1)/np.sqrt(len(dd))
print(f"  change in TOTAL opponent deaths per round: {dd.mean():+.4f} (t={dd.mean()/se:+.2f})"
      f"  -- i.e. the oracle does not kill more opponents, it re-attributes deaths")

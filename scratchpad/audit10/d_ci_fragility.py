#!/usr/bin/env python3
"""Audit 10 -- how stable is the 'CI excludes 0' verdict on the headline number?

analyze.py bootstraps with a hard-coded default_rng(12345), n_boot=10000. That makes
the printed interval reproducible, but it does NOT make it robust: the interval is a
Monte-Carlo estimate and its endpoint has its own sampling error. Redo the same
bootstrap with 200 other RNG seeds and count how often the lower bound clears 0.
Also give the normal-theory and the exact-sign-flip (permutation) answers, which have
no Monte-Carlo endpoint noise at all.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
CEIL = REPO / "scratchpad/strategy/ceil"


def col(path, name):
    rows = [r for r in csv.DictReader(open(path)) if r["code"] == "user_agent"]
    rows.sort(key=lambda r: int(r["round"]))
    return np.array([float(r[name]) for r in rows])


a = col(CEIL / "huntceil4k_k-1__task4_rb_ship990731.csv", "score")
b = col(CEIL / "huntceil4k_k4__task4_rb_ship990731.csv", "score")
d = b - a
n = len(d)
print(f"n={n}  mean diff = {d.mean():+.5f}  sd = {d.std(ddof=1):.4f}  se = {d.std(ddof=1)/np.sqrt(n):.5f}")

# exactly what analyze.py does
rng = np.random.default_rng(12345)
means = d[rng.integers(0, n, size=(10000, n))].mean(axis=1)
lo, hi = np.percentile(means, [2.5, 97.5])
print(f"analyze.py's own bootstrap (rng 12345, 10k):  [{lo:+.4f}, {hi:+.4f}]  -> "
      f"{'EXCLUDES 0' if lo > 0 else 'INCLUDES 0'}")

los, his = [], []
for s in range(200):
    r = np.random.default_rng(1000 + s)
    m = d[r.integers(0, n, size=(10000, n))].mean(axis=1)
    l, h = np.percentile(m, [2.5, 97.5])
    los.append(l); his.append(h)
los = np.array(los)
print(f"200 other bootstrap seeds: lower bound mean {los.mean():+.5f}, "
      f"sd {los.std():.5f}, range [{los.min():+.5f}, {los.max():+.5f}]")
print(f"   fraction of bootstrap seeds where the CI excludes 0: {(los > 0).mean():.1%}")

se = d.std(ddof=1) / np.sqrt(n)
print(f"normal-theory 95% CI: [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]  "
      f"t = {d.mean()/se:.3f}")
from math import erf, sqrt
p = 2 * (1 - 0.5 * (1 + erf(abs(d.mean() / se) / sqrt(2))))
print(f"two-sided p (normal approx): {p:.4f}")

# sign-flip permutation test on the paired differences
r = np.random.default_rng(7)
signs = r.choice([-1.0, 1.0], size=(200000, n))
null = (signs * d).mean(axis=1)
print(f"sign-flip permutation p: {(np.abs(null) >= abs(d.mean())).mean():.4f}")

# BCa-free sanity: studentised bootstrap lower bound
r = np.random.default_rng(99)
idx = r.integers(0, n, size=(20000, n))
bs = d[idx]
tstar = (bs.mean(axis=1) - d.mean()) / (bs.std(axis=1, ddof=1) / np.sqrt(n))
q = np.percentile(tstar, [2.5, 97.5])
print(f"studentised bootstrap CI: [{d.mean()-q[1]*se:+.4f}, {d.mean()-q[0]*se:+.4f}]")

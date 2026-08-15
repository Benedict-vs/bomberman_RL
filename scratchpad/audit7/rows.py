#!/usr/bin/env python3
"""Audit 7: is row 55060 a second fixed point, or an aliased mixture?

Decodes the decisive rows, and for each measures whether the candidate digit
(bucketed BFS distance to the nearest opponent) separates the fatal visits from
the safe ones *inside* the row. A proper permutation test replaces the max-
statistic used in analyse.py.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

with (HERE / "probe_ship.pkl").open("rb") as fh:
    d = pickle.load(fh)
rec, rounds = d["records"], d["rounds"]

q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
SIZES = (4, 4, 4, 4, 5, 5, 2, 5)
NB = {0: "blocked", 1: "lethal", 2: "in_blast", 3: "clear"}
DIRS = {0: "-", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}


def decode(row: int):
    out = []
    for size in reversed(SIZES):
        out.append(row % size)
        row //= size
    return tuple(reversed(out))


def describe(row: int) -> str:
    g = decode(row)
    return (f"nb=[{','.join(NB[v] for v in g[:4])}] own_danger={g[4]} "
            f"digit6={DIRS[g[5]]} bomb_useful={g[6]} dist_bucket={g[7]}")


def bucket(v: int) -> int:
    if v < 0:
        return 0
    if v <= 2:
        return 1
    if v <= 5:
        return 2
    return 3


BUCKET_NAME = {0: "unreachable", 1: "<=2", 2: "3-5", 3: ">=6"}

# label: died within k steps
by_round = defaultdict(list)
for i, r in enumerate(rec):
    by_round[r["round"]].append(i)
death4 = np.zeros(len(rec), dtype=bool)
death1 = np.zeros(len(rec), dtype=bool)
for rd in rounds:
    idx = by_round[rd["round"]]
    if rd["died"] and idx:
        for j in idx[-4:]:
            death4[j] = True
        death1[idx[-1]] = True

idx_by_row = defaultdict(list)
for i, r in enumerate(rec):
    idx_by_row[r["row"]].append(i)

rng = np.random.default_rng(20260816)


def perm_test(idxs, label, n_perm=20000):
    """Chi-square-ish: sum over buckets of n_b*(p_b - p)^2, permuted."""
    b = np.array([bucket(rec[i]["opp_dist"]) for i in idxs])
    y = label[idxs].astype(float)
    p = y.mean()

    def stat(yv):
        s = 0.0
        for v in np.unique(b):
            m = b == v
            s += m.sum() * (yv[m].mean() - p) ** 2
        return s

    obs = stat(y)
    null = np.array([stat(rng.permutation(y)) for _ in range(n_perm)])
    return obs, float((null >= obs).mean())


print("=== the rows the ledger argues about ===")
for row in (55060, 59160, 34110, 8720, 32160, 34210):
    idxs = idx_by_row.get(row, [])
    if not idxs:
        print(f"row {row}: not visited")
        continue
    qr = q[row]
    order = np.argsort(qr)[::-1]
    print(f"\nrow {row}  n={len(idxs)}  {describe(row)}")
    print("   Q: " + "  ".join(f"{ACTIONS[a]}={qr[a]:.3f}" for a in order[:3])
          + f"   gap={qr[order[0]]-qr[order[1]]:.3f}")
    b = np.array([bucket(rec[i]["opp_dist"]) for i in idxs])
    y4 = death4[idxs]
    print(f"   P(die<=4 | row) = {y4.mean():.4f}   ({y4.sum()} of {len(idxs)})")
    for v in sorted(set(b.tolist())):
        m = b == v
        print(f"     opp_dist {BUCKET_NAME[v]:12s} n={m.sum():5d}  "
              f"P(die<=4)={y4[m].mean():.4f}")
    if y4.sum() >= 3 and len(set(b.tolist())) > 1:
        obs, pv = perm_test(idxs, death4)
        print(f"   permutation test within this row: p = {pv:.4f}")

print("\n\n=== aggregate: pooled permutation test over all visited rows ===")
# Pool the within-row chi-square statistic; permute the candidate digit WITHIN
# each row so the null preserves every row's own death rate and digit marginal.
rows_used = [r for r, i in idx_by_row.items()
             if len(i) >= 30 and death4[i].sum() >= 2
             and len({bucket(rec[j]["opp_dist"]) for j in i}) > 1]
print(f"rows entering the pool: {len(rows_used)}  "
      f"steps: {sum(len(idx_by_row[r]) for r in rows_used)}")


def pooled(perm: bool) -> float:
    tot = 0.0
    for row in rows_used:
        idxs = idx_by_row[row]
        b = np.array([bucket(rec[i]["opp_dist"]) for i in idxs])
        y = death4[idxs].astype(float)
        if perm:
            y = rng.permutation(y)
        p = y.mean()
        for v in np.unique(b):
            m = b == v
            tot += m.sum() * (y[m].mean() - p) ** 2
    return tot


obs = pooled(False)
null = np.array([pooled(True) for _ in range(2000)])
print(f"pooled statistic obs = {obs:.2f}   null mean = {null.mean():.2f} "
      f"sd = {null.std():.2f}   p = {(null >= obs).mean():.4f}")

# How many deaths live in rows where the candidate digit is informative?
print("\n=== how much of the death mass is addressable ===")
tot_d = int(death4.sum())
addr = 0
for row in rows_used:
    idxs = idx_by_row[row]
    obs_r, pv = perm_test(idxs, death4, n_perm=4000)
    if pv < 0.05:
        addr += int(death4[idxs].sum())
print(f"deaths (die<=4 labelled steps) total {tot_d}; "
      f"in rows where the digit separates at p<0.05: {addr} ({addr/tot_d:.3f})")

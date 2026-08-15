#!/usr/bin/env python3
"""Audit 7: read probe_ship.pkl and answer the five questions in the report."""

from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

with (HERE / (sys.argv[1] if len(sys.argv) > 1 else "probe_ship.pkl")).open("rb") as fh:
    d = pickle.load(fh)

rec = d["records"]
rounds = d["rounds"]
n_rounds = len(rounds)

q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
valued = int(np.sum(np.any(q != 0.0, axis=1)))

print(f"=== sanity, n={n_rounds} rounds, seed {d['seed']} ===")
score = np.mean([r["score"] for r in rounds])
won = np.mean([1.0 if r["score"] == r["best"] else 0.0 for r in rounds])
print(f"score {score:.3f}  won {won:.3f}  died {np.mean([r['died'] for r in rounds]):.3f} "
      f"suicides {np.mean([r['suicides'] for r in rounds]):.3f} "
      f"kills {np.mean([r['kills'] for r in rounds]):.3f} "
      f"crates {np.mean([r['crates'] for r in rounds]):.2f}")
print(f"steps logged {len(rec)}  ({len(rec)/n_rounds:.1f}/round)")

print(f"\n=== coverage of the SHIPPED rung-4 table ===")
print(f"valued rows in table: {valued} / {q.shape[0]}")
visited = Counter(r["row"] for r in rec)
print(f"distinct rows visited by the greedy policy: {len(visited)}")
zero_steps = sum(1 for r in rec if r["zero"])
print(f"steps landing in an ALL-ZERO row: {zero_steps} / {len(rec)} = {zero_steps/len(rec):.4f}")
# E28 measured 2.6 % of alive steps on the solo-trained table.

# how concentrated is the visitation?
counts = np.array(sorted(visited.values())[::-1])
cum = np.cumsum(counts) / counts.sum()
for frac in (0.5, 0.9, 0.99):
    k = int(np.searchsorted(cum, frac)) + 1
    print(f"  {frac:.0%} of steps live in the top {k} rows")

print(f"\n=== digit 7 decomposition (E35's diagnosis) ===")
d7 = Counter()
for r in rec:
    if not r["have_bomb"]:
        d7["no bomb available"] += 1
    elif r["crate_in_blast"] and r["opp_in_blast"]:
        d7["digit7=1 : crate AND opponent"] += 1
    elif r["crate_in_blast"]:
        d7["digit7=1 : crate only"] += 1
    elif r["opp_in_blast"]:
        d7["digit7=1 : OPPONENT ONLY"] += 1
    else:
        d7["digit7=0 : nothing in blast"] += 1
for k, v in d7.most_common():
    print(f"  {k:34s} {v:7d}  {v/len(rec):.4f}")
opp_only = d7["digit7=1 : OPPONENT ONLY"]
both = d7["digit7=1 : crate AND opponent"]
crate_only = d7["digit7=1 : crate only"]
tot1 = opp_only + both + crate_only
if tot1:
    print(f"  --> of the steps where digit 7 = 1, {opp_only/tot1:.4f} are opponent-only,"
          f" {both/tot1:.4f} are both, {crate_only/tot1:.4f} crate-only")

print(f"\n=== candidate digit: BFS distance to nearest opponent ===")
od = np.array([r["opp_dist"] for r in rec])
print("distribution:", Counter(np.clip(od, -1, 8)).most_common())


def bucket(v: int) -> int:
    if v < 0:
        return 0          # no opponent reachable
    if v <= 2:
        return 1
    if v <= 5:
        return 2
    return 3


# Within-row variation: an appended digit only carries information if it VARIES
# inside the rows the policy occupies.
rows_by = defaultdict(list)
for r in rec:
    rows_by[r["row"]].append(bucket(r["opp_dist"]))
w = 0.0
tot = 0
pure = 0
for row, vals in rows_by.items():
    c = Counter(vals)
    n = len(vals)
    tot += n
    w += n * (1 - max(c.values()) / n)      # visit-weighted impurity
    if len(c) == 1:
        pure += n
print(f"visit-weighted within-row impurity of the bucketed opponent distance: {w/tot:.4f}")
print(f"steps in rows where the bucket never varies: {pure/tot:.4f}")

# Outcome lift: does the candidate digit separate deaths inside a row?
# Label each step by whether the agent died within the next 4 steps.
by_round = defaultdict(list)
for i, r in enumerate(rec):
    by_round[r["round"]].append(i)
death_soon = np.zeros(len(rec), dtype=bool)
for rd in rounds:
    idx = by_round[rd["round"]]
    if rd["died"] and idx:
        for j in idx[-4:]:
            death_soon[j] = True
base = death_soon.mean()
print(f"\nbase rate P(die within 4 steps) = {base:.4f}  (n={len(rec)})")

# within-row lift, visit-weighted over rows that both vary and see a death
num = 0.0
den = 0
detail = []
for row, idxs in defaultdict(list, {r: [] for r in rows_by}).items():
    pass
idx_by_row = defaultdict(list)
for i, r in enumerate(rec):
    idx_by_row[r["row"]].append(i)
for row, idxs in idx_by_row.items():
    if len(idxs) < 30:
        continue
    b = np.array([bucket(rec[i]["opp_dist"]) for i in idxs])
    dsoon = death_soon[idxs]
    if dsoon.sum() == 0 or len(set(b.tolist())) < 2:
        continue
    # best achievable split of this row on the candidate digit
    p_row = dsoon.mean()
    best = 0.0
    for v in set(b.tolist()):
        m = b == v
        if m.sum() < 5:
            continue
        best = max(best, abs(dsoon[m].mean() - p_row))
    num += len(idxs) * best
    den += len(idxs)
    detail.append((len(idxs), row, p_row, best))
print(f"visit-weighted within-row lift of the candidate digit on P(die<=4): {num/max(den,1):.4f}"
      f"  over {den} steps in {len(detail)} rows")
detail.sort(reverse=True)
print("  top rows by visits: (visits, row, P(death|row), best |lift| from the digit)")
for n, row, p, b in detail[:10]:
    print(f"    {n:6d}  row {row:6d}  p={p:.3f}  lift={b:.3f}")

# Same measurement for a *shuffled* candidate digit -- the null.
rng = np.random.default_rng(0)
num0 = 0.0
den0 = 0
for row, idxs in idx_by_row.items():
    if len(idxs) < 30:
        continue
    b = np.array([bucket(rec[i]["opp_dist"]) for i in idxs])
    b = rng.permutation(b)
    dsoon = death_soon[idxs]
    if dsoon.sum() == 0 or len(set(b.tolist())) < 2:
        continue
    p_row = dsoon.mean()
    best = 0.0
    for v in set(b.tolist()):
        m = b == v
        if m.sum() < 5:
            continue
        best = max(best, abs(dsoon[m].mean() - p_row))
    num0 += len(idxs) * best
    den0 += len(idxs)
print(f"  NULL (digit shuffled within row): {num0/max(den0,1):.4f}")

print(f"\n=== action gap on the shipped table ===")
gaps = np.array([r["gap"] for r in rec])
print(f"visit-weighted mean gap {gaps.mean():.3f}; share of decisions at gap < 1e-3: "
      f"{(gaps < 1e-3).mean():.4f}")

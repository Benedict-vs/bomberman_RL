"""Audit 12 / A6: where do the two arms' greedy policies actually diverge, and by how much?

A5 found argmax agreement between arms (0.986) is the same as within either arm (0.987),
so 'the mixed arm learned a different policy' is not visible in the table as a whole.
This asks whether the disagreements that ARE systematic (all 8 vs all 8) sit at large
value gaps -- a learned change -- or at the 1e-4..1e-2 margins this project has
repeatedly measured, where a uniform value compression flips the argmax without any
change in what the agent believes about death.
"""
from __future__ import annotations
import numpy as np

FS = (4, 4, 4, 4, 5, 5, 2, 5)
N = int(np.prod(FS))
ACT = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
idx = np.arange(N); rem = idx.copy(); digits = []
for base in reversed(FS):
    digits.append(rem % base); rem //= base
digits = list(reversed(digits))
own_danger, target, bomb_useful = digits[4], digits[5], digits[6]

parent = np.load("checkpoints/benedict_task4/q_table_parent.npy")
ctl = [np.load(f"checkpoints/benedict_task4/q_table_e37_PLB2_s{s}__ep20000.npy") for s in range(100, 108)]
mix = [np.load(f"checkpoints/benedict_task4/q_table_e42mix_s{s}__ep20000.npy") for s in range(200, 208)]

ct = np.array([np.any(q != parent, axis=1) for q in ctl])
mt = np.array([np.any(q != parent, axis=1) for q in mix])
both = ct.all(0) & mt.all(0)
rows = np.flatnonzero(both)

Ac = np.array([q[rows].argmax(1) for q in ctl])      # 8 x R
Am = np.array([q[rows].argmax(1) for q in mix])

# unanimous within each arm, and different between arms
unan_c = (Ac == Ac[0]).all(0)
unan_m = (Am == Am[0]).all(0)
systematic = unan_c & unan_m & (Ac[0] != Am[0])
print(f"common rows                              {len(rows)}")
print(f"unanimous within ctl                     {unan_c.sum()}")
print(f"unanimous within mix                     {unan_m.sum()}")
print(f"SYSTEMATIC flips (8/8 vs 8/8, different) {systematic.sum()}"
      f"   ({systematic.sum()/len(rows)*100:.2f} % of common rows)")

# margin at those rows: |Q(ctl argmax) - Q(mix argmax)| in each table
sel = np.flatnonzero(systematic)
gap_c, gap_m = [], []
for r in sel:
    row = rows[r]
    a_c, a_m = Ac[0, r], Am[0, r]
    gap_c.append(np.mean([q[row, a_c] - q[row, a_m] for q in ctl]))
    gap_m.append(np.mean([q[row, a_m] - q[row, a_c] for q in mix]))
gap_c, gap_m = np.array(gap_c), np.array(gap_m)
print(f"\nvalue gap defending each arm's own choice (mean over its 8 seeds):")
print(f"  ctl gap  median {np.median(gap_c):.4f}   mean {gap_c.mean():.4f}   "
      f"share < 0.05: {(gap_c < 0.05).mean():.2f}   < 0.5: {(gap_c < 0.5).mean():.2f}")
print(f"  mix gap  median {np.median(gap_m):.4f}   mean {gap_m.mean():.4f}   "
      f"share < 0.05: {(gap_m < 0.05).mean():.2f}   < 0.5: {(gap_m < 0.5).mean():.2f}")

danger = own_danger[rows] > 0
print(f"\nsystematic flips in danger rows: {int((systematic & danger).sum())} "
      f"of {int(danger.sum())} common danger rows")
print("action shift over systematic flips (ctl -> mix):")
from collections import Counter
c = Counter((ACT[Ac[0, r]], ACT[Am[0, r]]) for r in sel)
for k, v in c.most_common(12):
    print(f"   {k[0]:>5} -> {k[1]:<5}  {v}")

# The same comparison WITHIN the control arm, as the null: split ctl 4 vs 4.
print("\nNULL: split the control arm 4 vs 4 and repeat (same statistic, no treatment)")
import itertools
res = []
for combo in itertools.combinations(range(8), 4):
    other = [i for i in range(8) if i not in combo]
    A1, A2 = Ac[list(combo)], Ac[other]
    u1, u2 = (A1 == A1[0]).all(0), (A2 == A2[0]).all(0)
    res.append((u1 & u2 & (A1[0] != A2[0])).sum())
print(f"   systematic 4v4 flips within ctl: median {int(np.median(res))}, max {max(res)}"
      f"   (compare {systematic.sum()} for 8v8 ctl-vs-mix, a stricter test)")

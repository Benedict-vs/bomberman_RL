"""Audit 12 / A8: if there are no systematic argmax flips, where DOES the policy differ?

Per row, the fraction of the arm's 8 seeds choosing each action (untouched rows fall back
to the shared parent, which is identical for both arms). Rank rows by total-variation
distance between the arms and decode the top ones.
"""
from __future__ import annotations
import numpy as np
FS = (4, 4, 4, 4, 5, 5, 2, 5)
N = int(np.prod(FS)); ACT = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NB = {0: "blocked", 1: "LETHAL", 2: "in-blast", 3: "clear"}
D6 = {0: "none", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}

def decode(r):
    out = []
    for base in reversed(FS):
        out.append(r % base); r //= base
    return list(reversed(out))

parent = np.load("checkpoints/benedict_task4/q_table_parent.npy")
ctl = [np.load(f"checkpoints/benedict_task4/q_table_e37_PLB2_s{s}__ep20000.npy") for s in range(100, 108)]
mix = [np.load(f"checkpoints/benedict_task4/q_table_e42mix_s{s}__ep20000.npy") for s in range(200, 208)]

def dist(arms):
    P = np.zeros((N, 6))
    for q in arms:
        P[np.arange(N), q.argmax(1)] += 1
    return P / len(arms)

Pc, Pm = dist(ctl), dist(mix)
tv = 0.5 * np.abs(Pc - Pm).sum(1)
print(f"rows with any policy-distribution difference: {(tv > 0).sum()}")
print(f"  TV >= 0.5 (majority disagreement): {(tv >= 0.5).sum()}")
print(f"  TV == 1.0 (unanimous disagreement): {(tv == 1.0).sum()}")
ct = np.array([np.any(q != parent, axis=1) for q in ctl]).any(0)
mt = np.array([np.any(q != parent, axis=1) for q in mix]).any(0)
print(f"  of the TV==1 rows, touched by ctl: {int(((tv==1)&ct).sum())}, by mix: {int(((tv==1)&mt).sum())},"
      f" by both: {int(((tv==1)&ct&mt).sum())}")

order = np.argsort(-tv)[:20]
print(f"\ntop 20 rows by TV distance:")
print(f"{'row':>7}{'TV':>6}  n1 n2 n3 n4  d5 d6 d7 d8   ctl argmax(frac)          mix argmax(frac)")
for r in order:
    d = decode(int(r))
    ca = int(Pc[r].argmax()); ma = int(Pm[r].argmax())
    print(f"{r:7d}{tv[r]:6.2f}  {d[0]}  {d[1]}  {d[2]}  {d[3]}   {d[4]}  {d[5]}  {d[6]}  {d[7]}"
          f"   {ACT[ca]:<6}({Pc[r,ca]:.2f})            {ACT[ma]:<6}({Pm[r,ma]:.2f})")

# where does the action mass move overall, restricted to rows both arms touched?
both = ct & mt
print(f"\nnet action-mass shift over the {both.sum()} rows touched by both arms:")
for a in range(6):
    print(f"  {ACT[a]:<6} ctl {Pc[both,a].mean():.4f}   mix {Pm[both,a].mean():.4f}   "
          f"{Pm[both,a].mean()-Pc[both,a].mean():+.4f}")
d5 = np.array([decode(i)[4] for i in range(N)])
dang = both & (d5 > 0)
print(f"\nsame, danger rows only ({dang.sum()}):")
for a in range(6):
    print(f"  {ACT[a]:<6} ctl {Pc[dang,a].mean():.4f}   mix {Pm[dang,a].mean():.4f}   "
          f"{Pm[dang,a].mean()-Pc[dang,a].mean():+.4f}")

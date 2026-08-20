"""Audit 12 / A5: what did each arm actually write into the table?

E42's mechanism claim ('the agent learns that dying is not its fault and stops paying to
avoid it') is a claim about the LEARNED VALUE OF DEATH. It predicts, in the danger rows
(digit 5 > 0), that the mixed arm's Q-values are less negative and/or the gap between the
best and the worst action is smaller -- death is priced lower.

The competing explanation is coverage: the mixed arm simply updated fewer of the rows that
matter on a rule_based board, so its endgame policy is less converged.

Both are measured here against the shared warm-start parent.
"""
from __future__ import annotations
import numpy as np
from itertools import product

FS = (4, 4, 4, 4, 5, 5, 2, 5)
N = int(np.prod(FS))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# digit index -> value, for every row
idx = np.arange(N)
digits = []
rem = idx.copy()
for base in reversed(FS):
    digits.append(rem % base); rem //= base
digits = list(reversed(digits))          # digits[0..7]
own_danger = digits[4]
target = digits[5]
bomb_useful = digits[6]

parent = np.load("checkpoints/benedict_task4/q_table_parent.npy")
ctl = [np.load(f"checkpoints/benedict_task4/q_table_e37_PLB2_s{s}__ep20000.npy") for s in range(100, 108)]
mix = [np.load(f"checkpoints/benedict_task4/q_table_e42mix_s{s}__ep20000.npy") for s in range(200, 208)]

danger = own_danger > 0
safe = own_danger == 0
print(f"rows: {N}  danger rows {danger.sum()}  safe rows {safe.sum()}")

def stats(q):
    touched = np.any(q != parent, axis=1)
    out = {}
    out["rows touched"] = touched.sum()
    out["danger rows touched"] = (touched & danger).sum()
    out["safe rows touched"] = (touched & safe).sum()
    t = touched & danger
    out["danger minQ (touched)"] = q[t].min(axis=1).mean()
    out["danger maxQ (touched)"] = q[t].max(axis=1).mean()
    out["danger max-min (touched)"] = (q[t].max(axis=1) - q[t].min(axis=1)).mean()
    t2 = touched & safe
    out["safe minQ (touched)"] = q[t2].min(axis=1).mean()
    out["safe max-min (touched)"] = (q[t2].max(axis=1) - q[t2].min(axis=1)).mean()
    out["|q| mean over touched"] = np.abs(q[touched]).mean()
    return out

keys = list(stats(ctl[0]).keys())
print(f"\n{'quantity':<30}{'ctl mean':>12}{'sd':>9}{'mix mean':>12}{'sd':>9}{'diff':>10}")
for k in keys:
    a = np.array([stats(q)[k] for q in ctl]); b = np.array([stats(q)[k] for q in mix])
    print(f"{k:<30}{a.mean():12.3f}{a.std(ddof=1):9.3f}{b.mean():12.3f}{b.std(ddof=1):9.3f}{b.mean()-a.mean():+10.3f}")

# rows touched by BOTH arms in every seed, vs by one arm only
ctl_t = np.array([np.any(q != parent, axis=1) for q in ctl])
mix_t = np.array([np.any(q != parent, axis=1) for q in mix])
print(f"\nrows touched by >=1 ctl seed: {ctl_t.any(0).sum()}   by >=1 mix seed: {mix_t.any(0).sum()}")
print(f"rows touched by ALL 8 ctl:    {ctl_t.all(0).sum()}   by ALL 8 mix:    {mix_t.all(0).sum()}")
print(f"ctl-only (any): {(ctl_t.any(0) & ~mix_t.any(0)).sum()}   mix-only (any): {(mix_t.any(0) & ~ctl_t.any(0)).sum()}")

# policy agreement on the rows both arms touched in every seed
both = ctl_t.all(0) & mix_t.all(0)
print(f"\ncommon core rows (all 16 tables touched): {both.sum()}"
      f"   of which danger {int((both & danger).sum())}")
ag = []
for qa in ctl:
    for qb in mix:
        ag.append((qa[both].argmax(1) == qb[both].argmax(1)).mean())
within_c = [(ctl[i][both].argmax(1) == ctl[j][both].argmax(1)).mean()
            for i in range(8) for j in range(i + 1, 8)]
within_m = [(mix[i][both].argmax(1) == mix[j][both].argmax(1)).mean()
            for i in range(8) for j in range(i + 1, 8)]
print(f"argmax agreement  ctl-vs-mix {np.mean(ag):.3f}   within-ctl {np.mean(within_c):.3f}"
      f"   within-mix {np.mean(within_m):.3f}")

# the endgame rows on a rule_based board: no crates/coins left -> digit 6 points at an
# opponent, digit 7 says a bomb here catches one. Does the mixed arm choose BOMB more?
end = safe & (bomb_useful == 1) & (target > 0) & both
print(f"\nendgame-ish rows (safe, bomb_useful=1, target set, common): {end.sum()}")
for lab, arms in (("ctl", ctl), ("mix", mix)):
    frac = np.mean([(q[end].argmax(1) == 5).mean() for q in arms])
    print(f"  {lab}: P(argmax = BOMB) = {frac:.3f}")
for lab, arms in (("ctl", ctl), ("mix", mix)):
    d = danger & both
    frac = np.mean([(q[d].argmax(1) == 5).mean() for q in arms])
    fw = np.mean([(q[d].argmax(1) == 4).mean() for q in arms])
    # does the argmax follow digit 6 (the escape direction) in danger rows?
    follow = np.mean([(q[d].argmax(1) == (target[d] - 1)).mean() for q in arms])
    print(f"  {lab}: danger rows  P(BOMB)={frac:.3f}  P(WAIT)={fw:.3f}  P(follow digit6)={follow:.3f}")

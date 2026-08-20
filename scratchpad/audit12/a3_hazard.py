"""Audit 12 / A3: is the extra suicide a LATE-PHASE hazard, uniform across fields,
with the field-level differences explained by exposure to the late phase?

E42 attributes suicides +0.183 to the guard field alone and reads it as a global
devaluation of death. A2 showed the +0.183 is entirely post-step-200. This splits the
suicide rate into (a) hazard before 200, (b) probability of reaching 200, (c) hazard
after 200 conditional on reaching it -- computed identically on all three fields.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"
CUT = 200


def per_seed(arm, seed, field):
    f = EVAL / f"e42_{arm}_s{seed}__task4_{field}_val550731.csv"
    st, sui, kb = [], [], []
    for r in csv.DictReader(open(f)):
        if r["code"] != OURS: continue
        st.append(float(r["steps"])); sui.append(int(r["suicides"])); kb.append(int(r["killed_by_opponent"]))
    st, sui, kb = map(np.array, (st, sui, kb))
    reach = st >= CUT
    early_sui = ((sui == 1) & (st < CUT)).mean()
    p_reach = reach.mean()
    late_sui_haz = ((sui == 1) & (st >= CUT)).sum() / max(reach.sum(), 1)
    early_kb = ((kb == 1) & (st < CUT)).mean()
    late_kb_haz = ((kb == 1) & (st >= CUT)).sum() / max(reach.sum(), 1)
    return early_sui, p_reach, late_sui_haz, early_kb, late_kb_haz


def boot(a, b, seed=12345, n=10000):
    rng = np.random.default_rng(seed)
    d = [np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


names = ["suicide rate <200", "P(alive at 200)", "suicide hazard >=200 | alive",
         "killed_by rate <200", "killed_by hazard >=200 | alive"]
print(f"{'field':<10}{'quantity':<32}{'ctl':>8}{'mix':>8}{'diff':>9}{'95% CI':>22}")
for field in ("guard", "indist", "heldout"):
    A = np.array([per_seed("ctl", s, field) for s in range(100, 108)])
    B = np.array([per_seed("mix", s, field) for s in range(200, 208)])
    for i, nm in enumerate(names):
        lo, hi = boot(A[:, i], B[:, i])
        print(f"{field:<10}{nm:<32}{A[:,i].mean():8.3f}{B[:,i].mean():8.3f}"
              f"{B[:,i].mean()-A[:,i].mean():+9.3f}{f'[{lo:+.3f},{hi:+.3f}]':>22}")
    print()

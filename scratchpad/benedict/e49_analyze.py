#!/usr/bin/env python3
"""E48 scoring: the reward 2x2, against the pre-registration.

Unit of inference is the SEED (the pre-registered MDE of 0.35 comes from the between-seed
score SD of 0.249 at n=8). Bootstrap CI AND permutation p; disagreement => (fragile) =>
NOT demonstrated, per AGENTS.md.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"
CELLS = {
    "ctl": (lambda s, f: f"e42_ctl_s{s}__task4_{f}_val550731", list(range(100, 108))),
    "K":   (lambda s, f: f"e49_K_s{s}__task4_{f}_val550731",   list(range(500, 508))),
    "R":   (lambda s, f: f"e49_R_s{s}__task4_{f}_val550731",   list(range(510, 518))),
    "KR":  (lambda s, f: f"e49_KR_s{s}__task4_{f}_val550731",  list(range(520, 528))),
}
HIB = {"score": True, "coins": True, "kills": True, "crates": True,
       "coins_per_crate": True, "suicides": False, "survived": True, "invalid": False}


def seed_means(cell, field, metric):
    build, seeds = CELLS[cell]; out = []
    for s in seeds:
        f = EVAL / f"{build(s, field)}.csv"
        if not f.exists(): continue
        by = defaultdict(list)
        for r in csv.DictReader(open(f)): by[int(r["round"])].append(r)
        # coins_per_crate must be a ratio of TOTALS, not a mean of per-round ratios:
        # arm C leaves many rounds with zero crates, and per-round division blew up to 1e6.
        if metric == "coins_per_crate":
            co = sum(float(next(r for r in rows if r["code"] == OURS)["coins"]) for rows in by.values())
            cr = sum(float(next(r for r in rows if r["code"] == OURS)["crates"]) for rows in by.values())
            out.append(co / cr if cr > 0 else float("nan"))
        else:
            vals = [float(next(r for r in rows if r["code"] == OURS)[metric]) for rows in by.values()]
            out.append(float(np.mean(vals)))
    return np.array(out)


def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(np.mean(b) - np.mean(a))
    pool = np.concatenate([a, b]); na = len(a); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean() - pool[:na].mean()) >= obs: c += 1
    return c / n


def boot(a, b, seed=12345, n=10000):
    rng = np.random.default_rng(seed)
    d = [np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def row(a, b, label, hib=True):
    if len(a) == 0 or len(b) == 0:
        print(f"     {label:<26}(pending)"); return
    d = b.mean() - a.mean(); lo, hi = boot(a, b); p = perm_p(a, b)
    sig = lo * hi > 0; frag = sig != (p < 0.05)
    v = (("BESSER" if (d > 0) == hib else "SCHLECHTER") if (sig and not frag) else "nicht gezeigt")
    if frag: v += " (fragile)"
    print(f"     {label:<26}{a.mean():8.3f}{b.mean():8.3f}{d:+9.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:8.4f}  {v}")


for field in ("guard", "heldout"):
    name = "3x rule_based (GUARD, primary for P1)" if field == "guard" else "3x bindist_v2 (HELD OUT)"
    print(f"\n{'='*104}\n### {name}\n{'='*104}")
    print("  cell means:  " + "   ".join(
        f"{c} {seed_means(c, field, 'score').mean():.3f}" if len(seed_means(c, field, 'score')) else f"{c} --"
        for c in CELLS))
    for arm in ("K", "R", "KR"):
        print(f"\n  -- {arm} vs ctl")
        print(f"     {'metric':<26}{'ctl':>8}{'arm':>8}{'diff':>9}{'95% CI':>22}{'p':>8}  verdict")
        for m in HIB:
            row(seed_means("ctl", field, m), seed_means(arm, field, m), m, HIB[m])

print(f"\n{'='*104}\n### P4 scale vs balance: KR (ratio held, magnitude doubled) vs ctl\n{'='*104}")
for field in ("guard", "heldout"):
    c = {k: seed_means(k, field, "score") for k in CELLS}
    if all(len(v) for v in c.values()):
        inter = (c["KR"].mean() - c["K"].mean()) - (c["R"].mean() - c["ctl"].mean())
        print(f"  {field:<10} K {c['K'].mean():.3f}  R {c['R'].mean():.3f}  "
              f"KR {c['KR'].mean():.3f}  ctl {c['ctl'].mean():.3f}   "
              f"KR-ctl {c['KR'].mean()-c['ctl'].mean():+.3f} (P4 bar |.|<0.35)   interaction {inter:+.3f}")

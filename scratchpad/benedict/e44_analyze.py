#!/usr/bin/env python3
"""E44 scoring: the warm/scratch x rule_based/mixed 2x2, against the pre-registration.

Unit of inference is the SEED (the pre-registered MDE of 0.35 comes from the between-seed
score SD of 0.249 at n = 8). Every row carries a bootstrap CI AND a permutation p, and a row
where they disagree is marked (fragile) and counts as NOT demonstrated -- the AGENTS.md rule
that e42_analyze.py originally failed to apply.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"
CELLS = {                      # (label -> (filename-prefix builder, seeds))
    "warm_rb":  (lambda s, f: f"e42_ctl_s{s}__task4_{f}_val550731",    [100,101,102,103,104,105,106,107]),
    "warm_mix": (lambda s, f: f"e42_mix_s{s}__task4_{f}_val550731",    [200,201,202,203,204,205,206,207]),
    "scr_rb":   (lambda s, f: f"e44_scrrb_s{s}__task4_{f}_val550731",  [300,301,302,303,304,305,306,307]),
    "scr_mix":  (lambda s, f: f"e44_scrmix_s{s}__task4_{f}_val550731", [310,311,312,313,314,315,316,317]),
}
FIELDS = ["heldout", "indist", "guard"]


def seed_means(cell, field, metric):
    build, seeds = CELLS[cell]
    out = []
    for s in seeds:
        f = EVAL / f"{build(s, field)}.csv"
        if not f.exists():
            continue
        by = defaultdict(list)
        for r in csv.DictReader(open(f)):
            by[int(r["round"])].append(r)
        vals = []
        for rows in by.values():
            me = next(r for r in rows if r["code"] == OURS)
            opp = [float(r["score"]) for r in rows if r["code"] != OURS]
            if metric == "margin_mean": vals.append(float(me["score"]) - float(np.mean(opp)))
            elif metric == "margin_best": vals.append(float(me["score"]) - max(opp))
            else: vals.append(float(me[metric]))
        out.append(float(np.mean(vals)))
    return np.array(out)


def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(b) - np.mean(a)); pool = np.concatenate([a, b]); na = len(a); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean() - pool[:na].mean()) >= obs: c += 1
    return c / n


def boot(a, b, seed=12345, n=10000):
    rng = np.random.default_rng(seed)
    d = [np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(a, b, label, hib=True):
    if len(a) == 0 or len(b) == 0:
        print(f"  {label:<40}(missing)"); return None
    d = b.mean() - a.mean(); lo, hi = boot(a, b); p = perm_p(a, b)
    ci_sig = lo * hi > 0; fragile = ci_sig != (p < 0.05)
    v = (("BESSER" if (d > 0) == hib else "SCHLECHTER") if (ci_sig and not fragile)
         else "nicht gezeigt")
    if fragile: v += " (fragile)"
    print(f"  {label:<40}{a.mean():8.3f}{b.mean():8.3f}{d:+9.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:8.4f}  {v}")
    return d


print("=" * 108)
print("E44 -- the 2x2.  8 seeds/cell, 300 rounds, validation seed 550731")
print("=" * 108)

print("\n## Cell means (score / margin_mean), for orientation\n")
print(f"  {'cell':<12}" + "".join(f"{f:>26}" for f in FIELDS))
for cell in CELLS:
    row = f"  {cell:<12}"
    for f in FIELDS:
        sc = seed_means(cell, f, "score"); mg = seed_means(cell, f, "margin_mean")
        row += f"{(f'{sc.mean():.3f} / {mg.mean():+.3f}' if len(sc) else 'missing'):>26}"
    print(row)

print("\n## P2 GUARD -- is the scratch row even trained? bar = 3.07 on the guard field")
print("     (80 % of warm_rb's 3.832; if this fails, P1 is uninterpretable)\n")
for cell in ("scr_rb", "scr_mix"):
    v = seed_means(cell, "guard", "score")
    if len(v):
        print(f"  {cell:<12}guard score {v.mean():.3f}   "
              f"{'PASS' if v.mean() >= 3.07 else '*** FAIL ***'}   (per-seed {np.round(v,2)})")

print("\n## P1 PRIMARY -- the field effect from scratch, on the HELD-OUT field  (bar >= +0.35)\n")
print(f"  {'comparison':<40}{'A':>8}{'B':>8}{'diff':>9}{'95% CI':>22}{'perm p':>8}")
for m in ("margin_mean", "score", "margin_best", "survived", "suicides"):
    report(seed_means("scr_rb", "heldout", m), seed_means("scr_mix", "heldout", m),
           f"scr_mix - scr_rb  [{m}]", hib=(m != "suicides"))

print("\n## P3 INTERACTION -- did the warm start suppress the field effect?\n")
for f in FIELDS:
    dw = seed_means("warm_mix", f, "margin_mean").mean() - seed_means("warm_rb", f, "margin_mean").mean()
    ds = seed_means("scr_mix", f, "margin_mean").mean() - seed_means("scr_rb", f, "margin_mean").mean()
    print(f"  {f:<10} warm effect {dw:+.3f}   scratch effect {ds:+.3f}   difference {ds-dw:+.3f}")

print("\n## Scratch vs warm, same field -- how much is the parent worth?\n")
print(f"  {'comparison':<40}{'A':>8}{'B':>8}{'diff':>9}{'95% CI':>22}{'perm p':>8}")
for f in FIELDS:
    report(seed_means("warm_rb", f, "score"), seed_means("scr_rb", f, "score"),
           f"scr_rb - warm_rb  [score, {f}]")

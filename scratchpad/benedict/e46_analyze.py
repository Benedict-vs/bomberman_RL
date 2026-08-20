#!/usr/bin/env python3
"""E46 scoring against the pre-registration. Arena-paired (one fixed table, so the only
variance is evaluation noise). Bootstrap CI + sign-flip permutation p; a row where they
disagree is (fragile) and counts as NOT demonstrated."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

D = Path("scratchpad/benedict/e46")
OURS = "user_agent"


def load(v, fld):
    f = D / f"e46_v{v}__{fld}_ship990731.csv"
    if not f.exists(): return None
    by = defaultdict(list)
    for r in csv.DictReader(open(f)): by[int(r["round"])].append(r)
    return by


def series(by, m):
    out = {}
    for rd, rows in by.items():
        me = next(r for r in rows if r["code"] == OURS)
        opp = [float(r["score"]) for r in rows if r["code"] != OURS]
        out[rd] = (float(me["score"]) - float(np.mean(opp))) if m == "margin_mean" else float(me[m])
    return out


def signflip(d, n=20000):
    rng = np.random.default_rng(0); obs = abs(d.mean())
    null = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1)
    return float((np.abs(null) >= obs).mean())


def boot(d, seed=12345, n=4000):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


HIB = {"score": True, "margin_mean": True, "coins": True, "kills": True,
       "suicides": False, "survived": True, "crates": True, "bombs": True}

for fld in ("ext_xiaoxiae_binary_v6", "rule_based_agent"):
    ctl = load(0, fld)
    if ctl is None:
        print(f"\n### {fld}: control pending"); continue
    print(f"\n{'='*104}\n### field = 3x {fld}\n{'='*104}")
    c0 = {m: series(ctl, m) for m in HIB}
    print("  control: " + "  ".join(f"{m} {np.mean(list(c0[m].values())):.3f}" for m in ("score","margin_mean","suicides","survived")))
    for v in (5, 4, 3):
        arm = load(v, fld)
        if arm is None:
            print(f"\n  -- veto {v}: pending"); continue
        print(f"\n  -- veto {v} vs control")
        print(f"     {'metric':<14}{'ctl':>9}{'arm':>9}{'diff':>10}{'95% CI':>22}{'p':>9}   verdict")
        for m in HIB:
            a = series(arm, m); common = sorted(set(c0[m]) & set(a))
            d = np.array([a[r] - c0[m][r] for r in common])
            lo, hi = boot(d); p = signflip(d)
            sig = lo * hi > 0; frag = sig != (p < 0.05)
            verdict = (("BESSER" if (d.mean() > 0) == HIB[m] else "SCHLECHTER")
                       if (sig and not frag) else "nicht gezeigt")
            if frag: verdict += " (fragile)"
            print(f"     {m:<14}{np.mean([c0[m][r] for r in common]):9.3f}"
                  f"{np.mean([a[r] for r in common]):9.3f}{d.mean():+10.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:9.4f}   {verdict}")

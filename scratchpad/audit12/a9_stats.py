"""Audit 12 / A9: is the E42 headline row significant, or is it the n=8 percentile
bootstrap being too narrow?

e42_analyze.py's boot_ci resamples 8 seeds per arm and takes 2.5/97.5 percentiles. At
n=8 that interval is systematically narrower than a t interval on the same data (no
(n-1)/n correction, no t vs z). Every reported row is re-scored three ways.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy import stats

EVAL = Path("results/eval/task4_tournament"); OURS = "benedict_task4"

def seedmean(arm, s, field, m):
    f = EVAL / f"e42_{arm}_s{s}__task4_{field}_val550731.csv"
    by = {}
    for r in csv.DictReader(open(f)): by.setdefault(int(r["round"]), []).append(r)
    v = []
    for rows in by.values():
        me = next(r for r in rows if r["code"] == OURS)
        opp = [float(r["score"]) for r in rows if r["code"] != OURS]
        if m == "margin_mean": v.append(float(me["score"]) - np.mean(opp))
        elif m == "crates_per_bomb": v.append(float(me["crates"]) / max(float(me["bombs"]), 1e-9))
        else: v.append(float(me[m]))
    return float(np.mean(v))

def boot(a, b, seed=12345, n=20000):
    rng = np.random.default_rng(seed)
    d = [np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))) for _ in range(n)]
    return np.percentile(d, 2.5), np.percentile(d, 97.5)

def perm(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(b.mean() - a.mean())
    pool = np.concatenate([a, b]); na = len(a); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean() - pool[:na].mean()) >= obs: c += 1
    return c / n

ROWS = [("heldout", "margin_mean"), ("heldout", "score"), ("heldout", "survived"),
        ("heldout", "crates_per_bomb"), ("indist", "score"), ("indist", "margin_mean"),
        ("guard", "score"), ("guard", "suicides"), ("guard", "survived"),
        ("guard", "killed_by_opponent")]
print(f"{'field':<9}{'metric':<18}{'diff':>8}{'bootstrap CI (E42)':>24}"
      f"{'Welch t CI':>24}{'perm p':>9}{'t p':>9}  verdict")
for field, m in ROWS:
    a = np.array([seedmean("ctl", s, field, m) for s in range(100, 108)])
    b = np.array([seedmean("mix", s, field, m) for s in range(200, 208)])
    d = b.mean() - a.mean()
    lo, hi = boot(a, b)
    t = stats.ttest_ind(b, a, equal_var=False)
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    df = se**4 / (a.var(ddof=1)**2/(len(a)**2*(len(a)-1)) + b.var(ddof=1)**2/(len(b)**2*(len(b)-1)))
    tc = stats.t.ppf(0.975, df) * se
    p = perm(a, b)
    bsig = lo*hi > 0; tsig = abs(d) > tc
    v = "AGREE" if bsig == tsig == (p < .05) else "DISAGREE -> fragile"
    print(f"{field:<9}{m:<18}{d:+8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>24}"
          f"{f'[{d-tc:+.3f}, {d+tc:+.3f}]':>24}{p:9.4f}{t.pvalue:9.4f}  {v}")
    print(f"{'':<27}bootstrap width {hi-lo:.3f}   t width {2*tc:.3f}"
          f"   ratio {(hi-lo)/(2*tc):.2f}")

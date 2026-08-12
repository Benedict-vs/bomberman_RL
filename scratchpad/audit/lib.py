import csv, glob, os
import numpy as np

def load(path):
    with open(path) as fh:
        r = csv.DictReader(fh)
        cols = r.fieldnames
        rows = list(r)
    out = {}
    for c in cols:
        vals = [x[c] for x in rows]
        try:
            out[c] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[c] = np.array(vals)
    return out

def mean_ci(x, n_boot=20000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), np.nan, np.nan
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def t_ci(x):
    """Small-n: t-based CI, honest about n=5."""
    from math import sqrt
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); sd = x.std(ddof=1)
    tcrit = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}.get(n, 1.96)
    h = tcrit * sd / sqrt(n)
    return m, m-h, m+h, sd

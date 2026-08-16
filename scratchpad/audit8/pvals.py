"""Exact paired-t p-values over the n=5 runs, and multiplicity accounting."""
import numpy as np, csv, math

BASE = "results/eval/task4_tournament"
SEEDS = [100, 101, 102, 103, 104]
METRICS = ["score", "won", "crates", "suicides", "survived", "kills", "coins",
           "killed_by_opponent", "bombs", "invalid", "steps"]


def mean(label, seed, ep, m):
    f = f"{BASE}/{label}_s{seed}__ep{ep}__task4_rb_val550731.csv"
    with open(f) as fh:
        v = [float(r[m]) for r in csv.DictReader(fh) if r["agent"] == "benedict_task4"]
    return sum(v) / len(v)


def tp(x):
    """two-sided p from a paired t with df=4, via the exact t CDF."""
    x = np.asarray(x, float); n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    df = n - 1
    # regularized incomplete beta for the t distribution
    from math import lgamma, exp
    def betainc(a, b, z):
        # continued fraction (Lentz), adequate here
        if z <= 0: return 0.0
        if z >= 1: return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = exp(math.log(z) * a + math.log(1 - z) * b - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m_ = i // 2
            if i == 0: num = 1.0
            elif i % 2 == 0: num = (m_ * (b - m_) * z) / ((a + 2*m_ - 1) * (a + 2*m_))
            else: num = -((a + m_) * (a + b + m_) * z) / ((a + 2*m_) * (a + 2*m_ + 1))
            d = 1.0 + num * d
            if abs(d) < 1e-30: d = 1e-30
            d = 1.0 / d
            c = 1.0 + num / c
            if abs(c) < 1e-30: c = 1e-30
            f *= c * d
            if abs(1 - c*d) < 1e-12: break
        return front * (f - 1.0)
    z = df / (df + t*t)
    p = betainc(df/2, 0.5, z)
    return t, p


for ep in (5000, 20000):
    print(f"\n=== ep{ep} : paired t (df=4) vs e33 ctl ===")
    for arm in ("benedict_q_e36_OPP", "benedict_q_e36_PLB"):
        print(f" {arm}")
        for m in METRICS:
            d = np.array([mean(arm, s, ep, m) - mean("benedict_q_e33_ctl", s, ep, m)
                          for s in SEEDS])
            t, p = tp(d)
            sign = sum(d > 0)
            print(f"   {m:<20} d={d.mean():+8.4f}  t={t:+6.2f}  p={p:.4f}  signs +{sign}/5")

"""E37 core numbers: per-arm run means, paired contrasts, CIs, sign counts."""
from __future__ import annotations
import pickle, itertools, sys
import numpy as np
from scipy import stats

ROOT = "/Users/benedictvonschubert/Projects/bomberman_RL"
D = pickle.load(open(f"{ROOT}/scratchpad/audit9/e37.pkl", "rb"))
ARMS = ["ctl2", "PLB2", "PAR", "SHF"]
EPS = [5000, 10000, 20000]
SEEDS = list(range(100, 115))
METRICS = ["score", "won", "coins", "kills", "crates", "bombs", "suicides",
           "survived", "died", "killed_by_opponent", "steps", "invalid",
           "rank", "round_steps", "think_max_ms"]


def runmeans(arm, ep, metric):
    return np.array([D[(arm, s, ep)][metric].mean() for s in SEEDS])


def paired(a, b, ep, metric):
    """a - b, paired at run level over the 15 seeds."""
    x = runmeans(a, ep, metric) - runmeans(b, ep, metric)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h, p, int((x > 0).sum()), x


if __name__ == "__main__":
    print("=== per-arm run-mean ± 95% CI (over 15 seeds) ===")
    for metric in ["score", "won", "crates", "coins", "kills", "suicides", "survived"]:
        print(f"\n-- {metric}")
        print(f"{'ep':>6} " + " ".join(f"{a:>18}" for a in ARMS))
        for ep in EPS:
            cells = []
            for a in ARMS:
                v = runmeans(a, ep, metric)
                h = stats.t.ppf(0.975, 14) * v.std(ddof=1) / np.sqrt(15)
                cells.append(f"{v.mean():8.4f}±{h:7.4f}")
            print(f"{ep:>6} " + " ".join(f"{c:>18}" for c in cells))

    print("\n\n=== paired contrasts vs ctl2 ===")
    for metric in METRICS:
        print(f"\n-- {metric}")
        for a in ["PLB2", "PAR", "SHF"]:
            row = []
            for ep in EPS:
                m, lo, hi, p, npos, _ = paired(a, "ctl2", ep, metric)
                star = "*" if lo * hi > 0 else " "
                row.append(f"ep{ep//1000:>2}k {m:+8.4f} [{lo:+7.4f},{hi:+7.4f}] p={p:6.4f} {npos:2d}/15{star}")
            print(f"  {a:>5} | " + " | ".join(row))

    print("\n\n=== other contrasts (score, won, crates) at ep20000 ===")
    for metric in ["score", "won", "crates", "coins", "suicides", "survived"]:
        for a, b in [("PLB2", "SHF"), ("PLB2", "PAR"), ("PAR", "SHF")]:
            m, lo, hi, p, npos, _ = paired(a, b, 20000, metric)
            print(f"  {metric:>10} {a}-{b}: {m:+8.4f} [{lo:+7.4f},{hi:+7.4f}] p={p:6.4f} {npos}/15")

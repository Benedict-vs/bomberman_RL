"""Re-derive E36's numbers from the evaluation CSVs. Run-level pairing, n=5."""
import sys, glob, itertools
import numpy as np
import csv as _csv

BASE = "results/eval/task4_tournament"
SEEDS = [100, 101, 102, 103, 104]
METRICS = ["score", "won", "crates", "suicides", "survived", "kills", "coins",
           "died", "killed_by_opponent", "bombs", "invalid", "steps", "round_steps"]


def load(label, seed, ep):
    f = f"{BASE}/{label}_s{seed}__ep{ep}__task4_rb_val550731.csv"
    with open(f) as fh:
        rows = [r for r in _csv.DictReader(fh) if r["agent"] == "benedict_task4"]
    rows.sort(key=lambda r: int(r["round"]))
    return {m: np.array([float(r[m]) for r in rows]) for m in rows[0]
            if m not in ("agent", "code")}


def run_means(label, ep):
    """per-run mean of each metric, plus per-round matrices for arena pairing"""
    out = {}
    rounds = {}
    for s in SEEDS:
        d = load(label, s, ep)
        out[s] = {m: d[m].mean() for m in METRICS if m in d}
        rounds[s] = {m: d[m] for m in METRICS if m in d}
    return out, rounds


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    # t_{0.975, 4} = 2.776
    tcrit = {4: 2.776, 3: 3.182, 9: 2.262, 5: 2.571}[n - 1]
    return m, m - tcrit * se, m + tcrit * se


def report(ep, arms):
    ctl, ctl_r = run_means("benedict_q_e33_ctl", ep)
    print(f"\n==================== ep{ep} ====================")
    print(f"{'metric':<20}{'ctl':>9}", *[f"{a:>26}" for a in arms])
    for m in METRICS:
        if m not in ctl[SEEDS[0]]:
            continue
        cm = np.array([ctl[s][m] for s in SEEDS])
        row = f"{m:<20}{cm.mean():>9.4f}"
        for a in arms:
            am, _ = run_means(a, ep)
            av = np.array([am[s][m] for s in SEEDS])
            d, lo, hi = t_ci(av - cm)
            star = "*" if lo * hi > 0 else " "
            row += f"  {av.mean():>7.4f} {d:+.3f}[{lo:+.3f},{hi:+.3f}]{star}"
        print(row)


def per_seed(ep, arms, metric):
    ctl, _ = run_means("benedict_q_e33_ctl", ep)
    print(f"\n--- per-seed {metric} @ep{ep} ---")
    print(f"{'seed':<6}{'ctl':>9}", *[f"{a:>20}" for a in arms])
    for s in SEEDS:
        row = f"{s:<6}{ctl[s][metric]:>9.4f}"
        for a in arms:
            am, _ = run_means(a, ep)
            row += f"{am[s][metric]:>12.4f} ({am[s][metric]-ctl[s][metric]:+.3f})"
        print(row)


if __name__ == "__main__":
    arms = ["benedict_q_e36_OPP", "benedict_q_e36_PLB"]
    for ep in (5000, 20000):
        report(ep, arms)
    for m in ("crates", "score", "won", "suicides", "survived"):
        per_seed(20000, arms, m)
        per_seed(5000, arms, m)

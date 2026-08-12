import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load, t_ci, mean_ci

R = "/Users/benedictvonschubert/Projects/bomberman_RL/results/eval/task3_opponents/"
METRICS = ["score","coins","crates","bombs","survived","killed_by_opponent","suicides","kills","steps","invalid"]

def summ(path):
    d = load(path)
    ours = d["agent"] == "benedict_task2"
    if not ours.any():
        ours = np.array([a.startswith("benedict") for a in d["agent"]])
    out = {m: float(np.nanmean(d[m][ours])) for m in METRICS if m in d}
    out["n"] = int(ours.sum())
    return out

def arm(pattern):
    files = sorted(glob.glob(R + pattern))
    per = [summ(f) for f in files]
    return files, per

def show(name, files, per):
    print(f"\n--- {name}  ({len(files)} runs)")
    print(f"{'metric':<20}{'mean':>9}{'sd':>8}{'t-CI lo':>10}{'t-CI hi':>10}   per-run")
    for m in METRICS:
        if m not in per[0]: continue
        v = np.array([p[m] for p in per])
        mu, lo, hi, sd = t_ci(v) if len(v) > 1 else (v[0], np.nan, np.nan, np.nan)
        print(f"{m:<20}{mu:>9.3f}{sd:>8.3f}{lo:>10.3f}{hi:>10.3f}   " +
              " ".join(f"{x:.2f}" for x in v))

# floor
for lab, pat in [("FLOOR rung2ship @ cc/550731","benedict_q_rung2ship__task3_cc_val550731.csv"),
                 ("FLOOR rung2ship @ rb/550731","benedict_q_rung2ship__task3_rb_val550731.csv"),
                 ("E24 ship @ cc/20260731","benedict_q_e24_ship__task3_coin_collector.csv"),
                 ("E24 ship @ rb/20260731","benedict_q_e24_ship__task3_rule_based.csv"),
                 ("E24 ship @ peaceful/20260731","benedict_q_e24_ship__task3_peaceful.csv")]:
    f = R + pat
    if os.path.exists(f): show(lab, [f], [summ(f)])

for a in "AB":
    for ep in (20000, 40000):
        f, p = arm(f"benedict_q_e25_{a}_s1?__ep{ep}__task3_cc_val550731.csv")
        if f: show(f"ARM {a} @{ep} cc", f, p)
    f, p = arm(f"benedict_q_e25_{a}_s1?__ep40000__task3_rb_val550731.csv")
    if f: show(f"ARM {a} @40000 rb", f, p)

# paired B - A over run index
print("\n=== paired B - A (same BM_RUN_INDEX), n=5 runs, cc @40k ===")
for m in METRICS:
    va, vb = [], []
    for i in range(10, 15):
        va.append(summ(R + f"benedict_q_e25_A_s{i}__ep40000__task3_cc_val550731.csv")[m])
        vb.append(summ(R + f"benedict_q_e25_B_s{i}__ep40000__task3_cc_val550731.csv")[m])
    d = np.array(vb) - np.array(va)
    mu, lo, hi, sd = t_ci(d)
    print(f"{m:<20}{mu:>+9.3f}  [{lo:>+7.3f},{hi:>+7.3f}]")

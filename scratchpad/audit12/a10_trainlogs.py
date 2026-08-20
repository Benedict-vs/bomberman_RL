"""Audit 12 / A10: the training-stream premise of E42's mechanism.

E42 dismisses 'less experience' with '8 % fewer training steps' and offers instead:
against rule_based most deaths are self-inflicted, against strong agents most are
opponent-induced and unattributable. GOT_KILLED - KILLED_SELF is the opponent-induced
count, so the premise is directly checkable in the training logs.
"""
from __future__ import annotations
import csv, glob
import numpy as np

def load(paths):
    per = []
    for p in paths:
        R = list(csv.DictReader(open(p)))
        R = [r for r in R if int(r["episode"]) <= 20000]
        st = np.array([float(r["steps"]) for r in R])
        ks = np.array([float(r["KILLED_SELF"]) for r in R])
        gk = np.array([float(r["GOT_KILLED"]) for r in R])
        sr = np.array([float(r["SURVIVED_ROUND"]) for r in R])
        sc = np.array([float(r["score"]) for r in R])
        ep = np.array([int(r["episode"]) for r in R])
        last = ep > 19000
        per.append(dict(n_ep=len(R), steps_total=st.sum(), steps_mean=st.mean(),
                        killed_self=ks.mean(), got_killed=gk.mean(),
                        opp_killed=(gk - ks).mean(),
                        opp_share=(gk - ks).sum() / max(gk.sum(), 1),
                        survived=sr.mean(), score=sc.mean(),
                        score_last1k=sc[last].mean(), ks_last1k=ks[last].mean(),
                        opp_last1k=(gk - ks)[last].mean()))
    return per

ctl = load([f"results/train/task4_tournament/benedict_task3__q_e37_PLB2_s{s}.csv" for s in range(100, 108)])
mix = load([f"results/train/task4_tournament/benedict_task4__q_task4_s{s}.csv" for s in range(200, 208)])
keys = list(ctl[0].keys())
print(f"{'quantity':<18}{'ctl':>12}{'sd':>9}{'mix':>12}{'sd':>9}{'diff':>10}")
for k in keys:
    a = np.array([d[k] for d in ctl]); b = np.array([d[k] for d in mix])
    print(f"{k:<18}{a.mean():12.3f}{a.std(ddof=1):9.3f}{b.mean():12.3f}{b.std(ddof=1):9.3f}{b.mean()-a.mean():+10.3f}")
print("\nTotal training steps: ctl %.2f M   mix %.2f M   (%.1f %% fewer)" % (
    np.mean([d['steps_total'] for d in ctl])/1e6, np.mean([d['steps_total'] for d in mix])/1e6,
    100*(1-np.mean([d['steps_total'] for d in mix])/np.mean([d['steps_total'] for d in ctl]))))

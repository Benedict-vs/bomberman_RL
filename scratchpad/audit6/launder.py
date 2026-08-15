"""Does the 1-step bootstrap target see an imminent death?

Hypothesis (audit 6, H1 "bootstrap laundering"): the aggregated value function is
near-constant (E31: E[Q]=5.2+-0.5, corr(Q,G)=0.25), so for a step that leads to
death 2-4 steps later the target r + gamma*max_a' Q(row(s'),a') is
indistinguishable from a step that leads to safety. Only the terminal step itself
carries the -5. If true, 1-step Q-learning structurally cannot price the decision
that caused the death, and n-step / MC returns are the fix -- not a feature, not a
reward.

Prediction written before running: mean max_a Q(row_t) is flat at ~5.2 for every
k >= 2 steps before death and differs from the all-steps baseline by < 0.5.
Falsifier: it falls monotonically over the last 4 steps -> V does carry the
warning and the diagnosis is wrong.
"""
import pickle, sys, os, numpy as np
from collections import defaultdict

PKL = "scratchpad/audit5/deaths_e31_s80_ep20000.pkl"
TAB = "checkpoints/benedict_task4/q_table_e31_S0_s80__ep20000.npy"
K = 8

D = pickle.load(open(PKL, "rb"))
q = np.load(TAB)
steps = D["step_log"]
deaths = {d["round"]: d for d in D["deaths"]}

by = defaultdict(list)
for s in steps:
    by[s["round"]].append(s)
for v in by.values():
    v.sort(key=lambda z: z["step"])

valued = np.abs(q).sum(axis=1) > 0
maxq_all = q.max(axis=1)

# baseline over every alive step actually visited
allv = np.array([maxq_all[s["row"]] for s in steps])
print(f"alive steps {len(steps)}, rounds {len(by)}, deaths {len(deaths)}")
print(f"baseline  E[max_a Q(row)] over all alive steps = {allv.mean():.3f} "
      f"(sd {allv.std():.3f}, 5-95% {np.percentile(allv,5):.2f}-{np.percentile(allv,95):.2f})")

# survivors: rounds with no death, last K steps
surv_rounds = [r for r in by if r not in deaths]
print(f"rounds ending alive: {len(surv_rounds)}")

def profile(rounds, endstep):
    out = {}
    for k in range(K):
        vals = []
        for r in rounds:
            v = by[r]
            t = endstep(r) - k
            hit = [s for s in v if s["step"] == t]
            if hit:
                vals.append(maxq_all[hit[0]["row"]])
        out[k] = np.array(vals)
    return out

pd_ = profile(list(deaths), lambda r: deaths[r]["death_step"])
ps_ = profile(surv_rounds, lambda r: by[r][-1]["step"])

print("\nk = steps before the end.  mean max_a Q(row_{end-k})")
print(f"{'k':>3} {'died (n)':>12} {'mean':>8} {'sd':>6} | {'survived (n)':>13} {'mean':>8}")
for k in range(K):
    a, b = pd_[k], ps_[k]
    print(f"{k:>3} {len(a):>12} {a.mean():8.3f} {a.std():6.3f} | {len(b):>13} {b.mean():8.3f}")

# --- the decision that mattered: value of the taken action vs the row max
print("\nQ(row_t, action_t) on the same window (what the update pushes toward)")
def qprofile(rounds, endstep):
    for k in range(K):
        vals = []
        for r in rounds:
            v = by[r]
            t = endstep(r) - k
            hit = [s for s in v if s["step"] == t]
            if hit:
                s = hit[0]
                ai = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"].index(s["action"])
                vals.append(q[s["row"], ai])
        vals = np.array(vals)
        yield k, len(vals), vals.mean()
for k, n, m in qprofile(list(deaths), lambda r: deaths[r]["death_step"]):
    print(f"  k={k} n={n:4d} Q(taken)={m:.3f}")

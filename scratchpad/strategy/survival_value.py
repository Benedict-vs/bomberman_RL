"""Does staying alive earn score, and how much? And what is a kill worth on `won`?

Post-hoc on committed eval CSVs. Three questions:
  A. income rate -- score/coins per alive step, and what dying early forgoes
  B. the score -> won slope with a bootstrap CI (the conversion the ledger
     quotes as 0.113 was a ratio of means, not a marginal)
  C. a kill's value on `won`, split into the +5 we gain and the income we deny
"""
from __future__ import annotations

import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
US = "benedict_task4"
NUM = ("survived","round_steps","score","coins","kills","suicides","crates",
       "bombs","invalid","steps","died","killed_by_opponent","rank","won")

def load(p):
    out=[]
    with open(p) as fh:
        for r in csv.DictReader(fh):
            for k in NUM: r[k]=float(r[k])
            r["round"]=int(r["round"]); out.append(r)
    return out

def go(paths):
    us, opps = [], []
    for p in paths:
        by=defaultdict(list)
        for r in load(p): by[r["round"]].append(r)
        for k in sorted(by):
            rs=by[k]
            us.append([r for r in rs if r["code"]==US][0])
            opps.append([r for r in rs if r["code"]!=US])
    n=len(us)
    g=lambda k: np.array([m[k] for m in us])
    best=np.array([max(o["score"] for o in os_) for os_ in opps])
    steps, score, coins, kills, surv = g("steps"), g("score"), g("coins"), g("kills"), g("survived")
    rsteps = g("round_steps")
    print(f"pooled n={n} rounds from {len(paths)} evaluations")

    # A. income rate
    print("\n-- A. is time alive worth score?")
    print(f"  our alive steps: mean {steps.mean():.1f}  (round length {rsteps.mean():.1f})")
    d = surv==0
    print(f"  died rounds: alive {steps[d].mean():.1f} steps, coins {coins[d].mean():.3f}, "
          f"kills {kills[d].mean():.3f}")
    print(f"  survived   : alive {steps[~d].mean():.1f} steps, coins {coins[~d].mean():.3f}, "
          f"kills {kills[~d].mean():.3f}")
    print(f"  corr(alive steps, coins) = {np.corrcoef(steps, coins)[0,1]:+.3f}; "
          f"corr(alive steps, kills) = {np.corrcoef(steps, kills)[0,1]:+.3f}")
    # coins as a function of how long we lived, in deciles of alive steps
    q = np.quantile(steps, np.linspace(0,1,9))
    print("  coins / kills by alive-step octile (coins per 100 steps in brackets):")
    for i in range(8):
        m = (steps>=q[i]) & (steps<=q[i+1])
        if m.sum()<5: continue
        print(f"    steps {q[i]:6.0f}-{q[i+1]:6.0f}: n={int(m.sum()):4d} "
              f"coins {coins[m].mean():.3f} [{100*coins[m].mean()/steps[m].mean():.3f}] "
              f"kills {kills[m].mean():.3f} won {g('won')[m].mean():.3f} "
              f"bestopp {best[m].mean():.3f}")
    # marginal coin rate over the last stretch: coins earned by step t
    print("  NB: coins/step is not constant -- crates must be opened before coins exist.")

    # B. score -> won slope, bootstrap
    print("\n-- B. score -> won slope (recount, not a fit)")
    margin = score - best
    rng = np.random.default_rng(0)
    def slope(idx, d=1.0):
        m = margin[idx]
        return ((m+d>=0).mean() - (m>=0).mean())/d
    for d in (0.5, 1.0, 2.0):
        pt = slope(np.arange(n), d)
        bs = np.array([slope(rng.integers(0,n,n), d) for _ in range(2000)])
        lo,hi = np.percentile(bs,[2.5,97.5])
        print(f"  d=+{d}: dP(won) per point = {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  ratio-of-means the ledger quotes: won/score = {g('won').mean()/score.mean():.4f}")

    # C. what a kill is worth
    print("\n-- C. a kill on `won`")
    dead_sc=np.array([o["score"] for os_ in opps for o in os_ if o["died"]])
    live_sc=np.array([o["score"] for os_ in opps for o in os_ if not o["died"]])
    denial = live_sc.mean()-dead_sc.mean()
    print(f"  opponent score: died {dead_sc.mean():.3f} (n={len(dead_sc)}), "
          f"alive {live_sc.mean():.3f} (n={len(live_sc)}) -> upper bound on denial "
          f"{denial:.3f} pts, but a killed opponent was already dying in most rounds")
    # counterfactual: in each round, add 5 to us and (optionally) knock the leader down
    for label, dus, knock in (("+5 to us only", 5.0, False),
                              ("+5 to us AND leader loses the alive-vs-dead gap", 5.0, True)):
        m2 = margin + dus - (denial if knock else 0.0)*0  # placeholder
        pass
    # explicit: leader knocked to the 'died' mean
    knocked = np.array([np.sort([o["score"] for o in os_])[::-1] for os_ in opps], dtype=float)
    lead_removed = np.where(knocked[:,0] > dead_sc.mean(), np.maximum(knocked[:,1], dead_sc.mean()), knocked[:,0])
    base=(margin>=0).mean()
    print(f"  baseline won {base:.3f}")
    print(f"  +5 to us in every round             -> {(score+5-best>=0).mean():.3f}")
    print(f"  +5 to us AND we killed the leader   -> "
          f"{(score+5-np.maximum(lead_removed, 0)>=0).mean():.3f}")
    print(f"  (per +0.1 kills/round, linear: {0.1*((score+5-best>=0).mean()-base)*10/10:+.4f} won "
          f"-> {0.1*((score+5-best>=0).mean()-base):+.4f})")

    # what we would need
    print("\n-- D. what each +0.085 won costs in the two currencies")
    print(f"  +1.0 score. As coins: {coins.mean():.3f} -> {coins.mean()+1:.3f} per round "
          f"(fair share is 9/4 = 2.25; we already take {coins.mean()/2.25:.2f}x it)")
    print(f"  As kills: {kills.mean():.3f} -> {kills.mean()+0.2:.3f} per round")
    oppk=np.array([sum(o['kills'] for o in os_) for os_ in opps])
    oppd=np.array([sum(o['died'] for o in os_) for os_ in opps])
    oppsu=np.array([sum(o['suicides'] for o in os_) for os_ in opps])
    print(f"  pool: opponent deaths {oppd.mean():.3f}/round, own-bomb {oppsu.mean():.3f}, "
          f"kills credited to opponents {oppk.mean():.3f} (incl. killing us: "
          f"{g('killed_by_opponent').mean():.3f}) -> a rule_based agent kills "
          f"{oppk.mean()/3:.3f}/round vs our {kills.mean():.3f}")

if __name__ == "__main__":
    d = ROOT/"results/eval/task4_tournament"
    go([d/f for f in (sys.argv[1:] or [
        "benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv",
        "benedict_task4_shipped_e37__task4_rb_ship990731.csv"])])

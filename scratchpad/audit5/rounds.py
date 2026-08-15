import csv, glob, numpy as np, sys
def load(f):
    with open(f) as fh: return list(csv.DictReader(fh))
pat=sys.argv[1] if len(sys.argv)>1 else "results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep20000__task4_rb_val550731.csv"
me=[]
for f in sorted(glob.glob(pat)):
    me += [r for r in load(f) if r["agent"]=="benedict_task4"]
g=lambda k: np.array([float(r[k]) for r in me])
suic=g("suicides"); surv=g("survived"); kills=g("kills"); score=g("score")
coins=g("coins"); won=g("won"); crates=g("crates"); steps=g("steps"); kby=g("killed_by_opponent")
n=len(me); print("n rounds:",n)
print(f"overall  score {score.mean():.3f} won {won.mean():.3f} surv {surv.mean():.3f} suic {suic.mean():.3f} kby {kby.mean():.3f} kills {kills.mean():.3f} coins {coins.mean():.3f} crates {crates.mean():.2f}")
for name,mask in [("survived",surv==1),("suicide",suic==1),("killed_by_opp",kby==1)]:
    m=mask
    print(f"{name:14s} n={m.sum():5d} ({m.mean():.3f})  score {score[m].mean():6.3f}  won {won[m].mean():.3f}  kills {kills[m].mean():.3f}  coins {coins[m].mean():.3f}  crates {crates[m].mean():5.2f}  steps {steps[m].mean():6.1f}")
# what fraction of total score / kills comes from suicide rounds
print(f"\nshare of total score from suicide rounds: {score[suic==1].sum()/score.sum():.3f}")
print(f"share of total kills from suicide rounds: {kills[suic==1].sum()/max(kills.sum(),1):.3f}")
print(f"share of total wins from suicide rounds: {won[suic==1].sum()/won.sum():.3f}")
# kills in the same round as suicide -> mutual destruction?
print(f"P(kill>=1 | suicide) = {(kills[suic==1]>=1).mean():.3f} ; P(kill>=1 | survived) = {(kills[surv==1]>=1).mean():.3f}")
# steps histogram of death
print("\ndeath step (steps alive) percentiles for suicide rounds:", np.percentile(steps[suic==1],[10,25,50,75,90]).round(0))
print("steps percentiles for survived rounds:", np.percentile(steps[surv==1],[10,25,50,75,90]).round(0))

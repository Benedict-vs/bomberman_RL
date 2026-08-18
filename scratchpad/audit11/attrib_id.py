"""Audit 11: is the oracle's kill gain new deaths, or extra credit on the same deaths?

environment.py:238-263 credits EVERY explosion owner covering a victim, and adds KILLED_SELF
on top when one of those owners is the victim. So one death can simultaneously be the victim's
suicide and a +5 kill for us. Inclusion-exclusion over the E40 CSVs:

  X = (opp suicides) + (our kills) + (opp-on-opp kills) - (opp deaths)
    = the number of opponent deaths carrying more than one attribution.
If dX == d(our kills), every extra kill we are credited for is an extra multiply-tagged death,
not an extra death.
"""
import csv, numpy as np
from collections import defaultdict
from pathlib import Path
D = Path("scratchpad/benedict/e40")
def load(tag):
    br = defaultdict(list)
    with open(D / f"e40_{tag}__task4_rb_ship990731.csv") as fh:
        for row in csv.DictReader(fh): br[int(row["round"])].append(row)
    return br
def mine(rows, k): return float(next(r for r in rows if r["code"]=="user_agent")[k])
def oppsum(rows, k): return sum(float(r[k]) for r in rows if r["code"]!="user_agent")

def quantities(rows):
    D_  = oppsum(rows, "died")
    S   = oppsum(rows, "suicides")
    U   = mine(rows, "kills")
    us_killed_by_opp = mine(rows, "killed_by_opponent")
    O   = oppsum(rows, "kills") - us_killed_by_opp      # opponent blasts that hit an opponent
    return np.array([D_, S, U, O, S + U + O - D_])

NAMES = ["opp_deaths", "opp_suicides", "our_kills", "opp_on_opp_kills", "X (multi-attributed)"]
ctl = load("ctl")
A = np.array([quantities(ctl[r]) for r in sorted(ctl)])
print("arm       " + "".join(f"{n:>22s}" for n in NAMES))
print(f"{'ctl':9s} " + "".join(f"{v:22.4f}" for v in A.mean(axis=0)))
for tag in ("k4stale", "k4sim", "k8sim"):
    T = load(tag)
    B = np.array([quantities(T[r]) for r in sorted(T)])
    d = B - A
    se = d.std(axis=0, ddof=1)/np.sqrt(len(d))
    print(f"{tag:9s} " + "".join(f"{m:+9.4f}(t={m/s:+5.2f})" for m, s in zip(d.mean(axis=0), se)))
    print(f"{'':9s} " + " " * 66 + f"  d(our_kills)={d.mean(axis=0)[2]:+.4f}   dX={d.mean(axis=0)[4]:+.4f}")

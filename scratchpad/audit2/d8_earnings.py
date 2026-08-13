"""Gross earnings vs. gross cost of the reward stream, rung 2 vs rung 3.

The claim under test: the value function's dynamic range is set by the size of
the *earnings* stream relative to the action-independent step cost, and rung 3
loses most of it because nine coins are shared four ways.
"""
import sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scratchpad"/"audit"))
from lib import load

def mine(p, name="benedict_task3"):
    d = load(p); m = d['agent'] == name
    return {k: v[m] for k, v in d.items()}

def show(label, d, crate_r, coin_r=5.0, got_killed=-5.0, kill_r=0.0):
    st = d['steps'].mean()
    coins = coin_r * d['coins'].mean()
    crates = crate_r * d['crates'].mean()
    kills = kill_r * d['kills'].mean()
    step = -0.1 * st
    inval = -1.0 * d['invalid'].mean()
    death = got_killed * (1 - d['survived'].mean())
    earn = coins + crates + kills
    cost = step + inval + death
    print(f"{label:<38} steps {st:6.1f} | coin {coins:+7.2f} crate {crates:+7.2f} "
          f"kill {kills:+6.2f} = earn {earn:+7.2f} | step {step:+7.2f} inval {inval:+7.2f} "
          f"death {death:+6.2f} = cost {cost:+7.2f} | net {earn+cost:+7.2f}  "
          f"earn/cost {earn/abs(cost):5.2f}")

E2 = ROOT/"results/eval/task2_crates"; E3 = ROOT/"results/eval/task3_opponents"
# rung 2, the shipped table alone on classic
c = sorted(glob.glob(str(E2/"benedict_q_e23*ep20000__task2.csv")))
if c:
    show("rung 2 solo (E23 ship-lineage) crate .3", mine(c[0]), 0.3)
    show("rung 2 solo                     crate 1", mine(c[0]), 1.0)
print()
for f, lab in [("benedict_q_e26_F_huntoff__task3_cc_ship990731.csv", "rung 3 cc, F HUNT off"),
               ("benedict_q_e26_F_frozen__task3_cc_ship990731.csv",  "rung 3 cc, F HUNT on"),
               ("benedict_q_e26_F_frozen__task3_rb_ship990731.csv",  "rung 3 rb, F HUNT on")]:
    d = mine(str(E3/f))
    show(lab + "  crate .3", d, 0.3)
    show(lab + "  crate 1 ", d, 1.0)

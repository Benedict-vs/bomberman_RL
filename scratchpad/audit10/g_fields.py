#!/usr/bin/env python3
"""Audit 10 -- claim group B: the four-field robustness table.

B1 "kills scale inversely with the field's evasion quality"
B2 "82 % of our own-bomb deaths are opponent-induced"

Both are read off per-round marginals. Reread the same CSVs with the *pool* of killable
opponents and the *exposure* made explicit, and with the death-attribution identity checked.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
F = REPO / "scratchpad/strategy/fields"
EVAL = REPO / "results/eval/task4_tournament"

FIELDS = {
    "peaceful": F / "ship_e37__task4_field_peaceful.csv",
    "coin_coll": F / "ship_e37__task4_field_coin_collector.csv",
    "mixed": F / "ship_e37__task4_field_mixed.csv",
    "rule_based": EVAL / "benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv",
}

rng = np.random.default_rng(31337)


def rounds_of(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(int(r["round"]), []).append(r)
    return [by[k] for k in sorted(by)]


def f(r, key):
    return float(r[key])


print(f"{'field':11s} {'n':>4s} {'ourkills':>8s} {'oursteps':>8s} {'oppdeath':>8s} "
      f"{'oppsuic':>8s} {'takeable':>8s} {'ourshare':>8s} {'kills/1e3 opp-alive-steps':>26s}")
summary = {}
for name, path in FIELDS.items():
    rds = rounds_of(path)
    n = len(rds)
    mine_k, mine_steps, mine_bombs, mine_suic, mine_died = [], [], [], [], []
    opp_deaths, opp_suic, opp_steps, opp_kills = [], [], [], []
    dbl = []
    for rs in rds:
        me = [r for r in rs if r["code"] == "benedict_task4"][0]
        op = [r for r in rs if r["code"] != "benedict_task4"]
        mine_k.append(f(me, "kills")); mine_steps.append(f(me, "steps"))
        mine_bombs.append(f(me, "bombs")); mine_suic.append(f(me, "suicides"))
        mine_died.append(f(me, "died"))
        opp_deaths.append(sum(f(r, "died") for r in op))
        opp_suic.append(sum(f(r, "suicides") for r in op))
        opp_steps.append(sum(f(r, "steps") for r in op))
        opp_kills.append(sum(f(r, "kills") for r in op))
        tot_died = sum(f(r, "died") for r in rs)
        tot_cred = sum(f(r, "kills") + f(r, "suicides") for r in rs)
        dbl.append(tot_cred - tot_died)
    A = {k: np.array(v, float) for k, v in dict(
        mine_k=mine_k, mine_steps=mine_steps, mine_bombs=mine_bombs, mine_suic=mine_suic,
        mine_died=mine_died, opp_deaths=opp_deaths, opp_suic=opp_suic, opp_steps=opp_steps,
        opp_kills=opp_kills, dbl=dbl).items()}
    summary[name] = A
    takeable = A["opp_deaths"] - A["opp_suic"]     # deaths not self-inflicted
    share = A["mine_k"].sum() / max(takeable.sum(), 1e-9)
    rate = 1000 * A["mine_k"].sum() / A["opp_steps"].sum()
    print(f"{name:11s} {n:4d} {A['mine_k'].mean():8.3f} {A['mine_steps'].mean():8.1f} "
          f"{A['opp_deaths'].mean():8.3f} {A['opp_suic'].mean():8.3f} {takeable.mean():8.3f} "
          f"{share:8.1%} {rate:26.3f}")

print("\n### B1 -- the killable pool, not the evasion quality")
for name, A in summary.items():
    takeable = A["opp_deaths"] - A["opp_suic"]
    print(f"  {name:11s} opponents die {A['opp_deaths'].mean():.3f}/round of which "
          f"{A['opp_suic'].mean():.3f} by their own bomb -> only "
          f"{takeable.mean():.3f} deaths are available to anybody else; we take "
          f"{A['mine_k'].mean():.3f} = {A['mine_k'].sum()/max(takeable.sum(),1e-9):.1%}")

print("\n### death-attribution identity: sum(kills)+sum(suicides) - sum(died) per round")
print("    (>0 means a single death was credited twice: own blast AND an opponent's blast)")
for name, A in summary.items():
    d = A["dbl"]
    print(f"  {name:11s} mean excess {d.mean():+.4f}/round, positive in {(d > 0).mean():.1%} "
          f"of rounds, max {d.max():.0f}")

print("\n### B2 -- normalising our own-bomb deaths")
for name, A in summary.items():
    print(f"  {name:11s} suicides/round {A['mine_suic'].mean():.3f}  "
          f"per 1000 of our steps {1000*A['mine_suic'].sum()/A['mine_steps'].sum():.3f}  "
          f"per 100 of our bombs {100*A['mine_suic'].sum()/A['mine_bombs'].sum():.3f}  "
          f"died {A['mine_died'].mean():.3f}  suicide share of our deaths "
          f"{A['mine_suic'].sum()/max(A['mine_died'].sum(),1e-9):.1%}")

print("\n### how many opponent bombs are on the board at all? (bombs placed by opponents/round)")
for name, path in FIELDS.items():
    rds = rounds_of(path)
    ob = np.array([sum(f(r, "bombs") for r in rs if r["code"] != "benedict_task4") for rs in rds])
    oc = np.array([sum(f(r, "crates") for r in rs if r["code"] != "benedict_task4") for rs in rds])
    mc = np.array([f([r for r in rs if r["code"] == "benedict_task4"][0], "crates") for rs in rds])
    print(f"  {name:11s} opponent bombs {ob.mean():6.2f}/round   opponent crates {oc.mean():6.2f}"
          f"   our crates {mc.mean():6.2f}   total crates cleared {(oc+mc).mean():6.2f}")

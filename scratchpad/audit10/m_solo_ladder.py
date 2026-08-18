#!/usr/bin/env python3
"""Audit 10 -- B2: decompose the own-bomb death rate into (a) nothing, (b) bodies that
never bomb, (c) bodies that bomb. The 'in isolation' arm the claim asserts but never ran
is scratchpad/audit10/audit10_solo_noopp.csv (tools/evaluate.py, 300 rounds, seed 990731)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
ARMS = {
    "no opponents":            REPO / "scratchpad/audit10/audit10_solo_noopp.csv",
    "3x peaceful (no bombs)":  REPO / "scratchpad/strategy/fields/ship_e37__task4_field_peaceful.csv",
    "3x coin_collector":       REPO / "scratchpad/strategy/fields/ship_e37__task4_field_coin_collector.csv",
    "mixed (1 rule_based)":    REPO / "scratchpad/strategy/fields/ship_e37__task4_field_mixed.csv",
    "3x rule_based":           REPO / "results/eval/task4_tournament/benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv",
}
print(f"{'field':24s}{'n':>5s}{'suic/rd':>9s}{'/1e3 step':>11s}{'/100 bomb':>11s}"
      f"{'oursteps':>10s}{'ourbombs':>10s}{'oppbombs':>10s}{'survived':>10s}")
base = None
for name, path in ARMS.items():
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(int(r["round"]), []).append(r)
    n = len(by)
    me, ob = [], []
    for k in sorted(by):
        rs = by[k]
        mine = [r for r in rs if r["code"] == "benedict_task4"][0]
        me.append((float(mine["suicides"]), float(mine["steps"]), float(mine["bombs"]),
                   float(mine["survived"])))
        ob.append(sum(float(r["bombs"]) for r in rs if r["code"] != "benedict_task4"))
    a = np.array(me); ob = np.array(ob)
    per_rd = a[:, 0].mean()
    per_step = 1000 * a[:, 0].sum() / a[:, 1].sum()
    per_bomb = 100 * a[:, 0].sum() / a[:, 2].sum()
    if base is None:
        base = (per_rd, per_step, per_bomb)
    print(f"{name:24s}{n:5d}{per_rd:9.3f}{per_step:11.3f}{per_bomb:11.3f}"
          f"{a[:,1].mean():10.1f}{a[:,2].mean():10.2f}{ob.mean():10.2f}{a[:,3].mean():10.3f}")

print("\nAttribution of the own-bomb death rate (per 100 of our own bombs):")
solo, peace, rb = None, None, None
for name, path in ARMS.items():
    rows = list(csv.DictReader(open(path)))
    mine = [r for r in rows if r["code"] == "benedict_task4"]
    v = 100 * sum(float(r["suicides"]) for r in mine) / sum(float(r["bombs"]) for r in mine)
    if name == "no opponents": solo = v
    if name.startswith("3x peaceful"): peace = v
    if name == "3x rule_based": rb = v
print(f"  our escape logic alone (empty board)         {solo:.3f}   "
      f"{solo/rb:6.1%} of the rule_based rate")
print(f"  + three bodies that block but never bomb     {peace:.3f}   "
      f"(+{peace-solo:.3f}, {100*(peace-solo)/(rb-solo):.1f} % of the gap)")
print(f"  + those bodies also bombing (3x rule_based)  {rb:.3f}   "
      f"(+{rb-peace:.3f}, {100*(rb-peace)/(rb-solo):.1f} % of the gap)")

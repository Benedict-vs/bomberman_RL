"""Audit 11 -- who actually kills the opponents when the oracle is switched on?

E40 reads +0.019 kills / +0.163 margin_best as "positioning to trap opponents".
The E40 CSVs already show opponent *suicides* rising (+0.035) while opponent
deaths-by-someone-else stay flat (-0.009), which is not what trapping looks
like. environment.py:238-263 credits EVERY explosion owner covering the victim,
so one death can be a suicide AND a kill for us at the same time -- the CSV
cannot separate those. This wraps `evaluate_explosions` (in this script only,
no framework file is touched) and records, for every death, the exact set of
blast owners.

usage: c_attrib.py <k> <trap_model> <rounds>
"""
from __future__ import annotations
import os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s
from environment import BombeRLeWorld, WorldArgs
import agent_code.benedict_task4.callbacks as cb
from hunt_ceiling_v2 import HuntCeiling

K = int(sys.argv[1]); MODEL = sys.argv[2]; ROUNDS = int(sys.argv[3])
SEED = 990731

q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = HuntCeiling(q, K, simultaneous=(MODEL == "sim"))
log_dir = REPO / "logs" / "audit11"; log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                 match_name=f"audit11_attrib_k{K}_{MODEL}", seed=SEED,
                 silence_errors=False, scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None
MYNAME = [a.name for a in world.agents if a.code_name == "user_agent"][0]

orig_eval = world.evaluate_explosions
c: Counter = Counter()

def patched():
    hits: dict[str, set[str]] = {}
    for exp in world.explosions:
        if exp.is_dangerous():
            for a in world.active_agents:
                if (not a.dead) and (a.x, a.y) in exp.blast_coords:
                    hits.setdefault(a.name, set()).add(exp.owner.name)
    orig_eval()
    for victim, owners in hits.items():
        who = set()
        if victim in owners:
            who.add("own")
        if MYNAME in owners and victim != MYNAME:
            who.add("us")
        if any(o not in (victim, MYNAME) for o in owners):
            who.add("opp")
        tag = "+".join(sorted(who)) or "none"
        side = "me" if victim == MYNAME else "opp"
        c[f"{side}:{tag}"] += 1
        c[f"{side}:deaths"] += 1

world.evaluate_explosions = patched

for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r)
    np.random.seed(SEED + r)
    world.new_round()
    while world.running:
        alive = [a for a in world.active_agents if a.code_name == "user_agent"]
        action = "WAIT"
        if alive:
            gs = world.get_state_for_agent(alive[0])
            if gs is not None:
                action = policy.act(gs)
        world.do_step(action)
    if (r + 1) % 250 == 0:
        print(f"  {r+1}/{ROUNDS}", flush=True)
world.end()

print(f"\nk={K} model={MODEL} rounds={ROUNDS}  (per round)")
for key in sorted(c):
    if ":" in key:
        print(f"  {key:<22} {c[key]:>6}  {c[key]/ROUNDS:>8.4f}")
us_credit = sum(v for k_, v in c.items() if k_.startswith("opp:") and "us" in k_.split(":")[1].split("+"))
us_credit_own = sum(v for k_, v in c.items()
                    if k_.startswith("opp:") and "us" in k_.split(":")[1].split("+")
                    and "own" in k_.split(":")[1].split("+"))
print(f"\n  opponent deaths we are credited for : {us_credit}  ({us_credit/ROUNDS:.4f}/round)")
print(f"    ... victim was ALSO in its own blast: {us_credit_own} "
      f"({us_credit_own/max(us_credit,1):.1%})")

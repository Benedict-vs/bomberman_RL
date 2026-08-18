"""Why does ext_aielka_ql_atom produce so many invalid actions?

Hypothesis to test (formed from its code, q_learning.py:37):
    `if state not in self.q_table or random.uniform(0,1) <= epsilon: return random.choice(ACTIONS)`
With epsilon forced to 0 (frozen opponent), the only way to hit that branch is a state that is
absent from the 329-entry table. A uniform random action from a 6-action set is invalid whenever
it walks into a wall/crate or bombs without a bomb available -- so the miss rate should predict
the invalid rate. Measured here by counting table misses during real play.
"""
import os, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np
from environment import BombeRLeWorld, WorldArgs
import agent_code.ext_aielka_ql_atom.q_learning as qlmod

CALLS = {"n": 0, "miss": 0}
_orig = qlmod.QLearningAgent.act
def counting_act(self, feature_vector, n_round, train=True):
    CALLS["n"] += 1
    if tuple(feature_vector) not in self.q_table:
        CALLS["miss"] += 1
    return _orig(self, feature_vector, n_round=n_round, train=train)
qlmod.QLearningAgent.act = counting_act

args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
                 replay=None, make_video=False, continue_without_training=True, log_dir="logs",
                 save_stats=False, match_name=None, seed=None, silence_errors=True, scenario="classic")
names = ["ext_aielka_ql_atom"] + ["rule_based_agent"] * 3
world = BombeRLeWorld(args, [(n, False) for n in names])
invalid = 0
for i in range(10):
    world.rng = np.random.default_rng(990731 + i)
    random.seed(990731 + i)
    world.new_round()
    while world.running:
        world.do_step()
    invalid += world.agents[0].invalid_actions if hasattr(world.agents[0], "invalid_actions") else 0
world.end()
n, miss = CALLS["n"], CALLS["miss"]
print(f"act() calls: {n}   table misses: {miss}  ({100*miss/n:.1f}% of steps)")
print(f"table size: 329 states")
print(f"expected invalid rate if every miss picks uniformly from 6 actions: see notes")

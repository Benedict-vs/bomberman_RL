"""Follow-up: the misses hypothesis was refuted (0/3384). Where do the invalid actions come from?

Records, per step, the state key, the greedy action, how many actions tie for the max, and
whether the environment then logged INVALID_ACTION for the agent.
"""
import os, sys, random, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np
import events as e
from environment import BombeRLeWorld, WorldArgs
import agent_code.ext_aielka_ql_atom.q_learning as qlmod

LOG = []
_orig = qlmod.QLearningAgent.act
def tracing_act(self, feature_vector, n_round, train=True):
    st = tuple(feature_vector)
    row = self.q_table.get(st)
    a = _orig(self, feature_vector, n_round=n_round, train=train)
    ties = sum(1 for v in row if v == max(row)) if row else -1
    LOG.append({"state": st, "action": a, "ties": ties, "invalid": None})
    return a
qlmod.QLearningAgent.act = tracing_act

args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
                 replay=None, make_video=False, continue_without_training=True, log_dir="logs",
                 save_stats=False, match_name=None, seed=None, silence_errors=True, scenario="classic")
world = BombeRLeWorld(args, [(n, False) for n in ["ext_aielka_ql_atom"] + ["rule_based_agent"]*3])
per_round = []
for i in range(40):
    world.rng = np.random.default_rng(990731 + i); random.seed(990731 + i)
    world.new_round()
    start, inv = len(LOG), 0
    while world.running:
        n_before = len(LOG)
        world.do_step()
        ev = world.agents[0].events
        if e.INVALID_ACTION in ev and len(LOG) > n_before:
            LOG[n_before]["invalid"] = True; inv += 1
        elif len(LOG) > n_before:
            LOG[n_before]["invalid"] = False
    rr = [r for r in LOG[start:] if r["invalid"] is not None]
    best_run = cur = 0
    for r in rr:
        cur = cur + 1 if r["invalid"] else 0
        best_run = max(best_run, cur)
    per_round.append((world.step, inv, best_run))
world.end()

rows = [r for r in LOG if r["invalid"] is not None]
inv = [r for r in rows if r["invalid"]]
print(f"steps={len(rows)}  invalid={len(inv)} ({100*len(inv)/len(rows):.1f}%)")
print("per round (steps, invalid, longest_consecutive_invalid_run):")
[print("   ", x) for x in sorted(per_round, key=lambda t: -t[1])[:8]]
print("\ninvalid actions by action:", collections.Counter(r["action"] for r in inv).most_common())
print("invalid actions by #tied-max in the row:", collections.Counter(r["ties"] for r in inv).most_common())
print("all steps by #tied-max:", collections.Counter(r["ties"] for r in rows).most_common())
print("\ndistinct states that ever produced an invalid action:",
      len({r["state"] for r in inv}), "of", len({r["state"] for r in rows}), "states visited")
top = collections.Counter(r["state"] for r in inv).most_common(3)
for stt, c in top:
    tot = sum(1 for r in rows if r["state"] == stt)
    acts = collections.Counter(r["action"] for r in rows if r["state"] == stt)
    print(f"  {c}/{tot} invalid  ties={[r['ties'] for r in rows if r['state']==stt][0]}  actions={dict(acts)}\n    state={stt}")
# longest run of consecutive invalid steps
best = cur = 0
for r in rows:
    cur = cur + 1 if r["invalid"] else 0
    best = max(best, cur)
print(f"\nlongest run of consecutive invalid steps: {best}")

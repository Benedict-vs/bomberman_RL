"""Independent check of E24: all-zero Q rows at the fatal step vs base rate."""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environment import BombeRLeWorld, WorldArgs   # noqa: E402
import events as e                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--opponents", default="coin_collector_agent")
ap.add_argument("--n-opponents", type=int, default=3)
ap.add_argument("--n-rounds", type=int, default=100)
ap.add_argument("--seed", type=int, default=20260731)
ap.add_argument("--window", type=int, default=4)
a = ap.parse_args()

cb = importlib.import_module("agent_code.benedict_task3.callbacks")
q = np.load(cb.MODEL_FILE)
zero = ~q.any(axis=1)
log_dir = ROOT / "logs" / "audit_death"; log_dir.mkdir(parents=True, exist_ok=True)
wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                  save_replay=False, replay=None, make_video=False,
                  continue_without_training=True, log_dir=str(log_dir),
                  save_stats=False, match_name="ad", seed=None,
                  silence_errors=False, scenario="classic")
line_up = ["benedict_task3"] + [a.opponents] * a.n_opponents if a.n_opponents else ["benedict_task3"]
world = BombeRLeWorld(wargs, [(n, False) for n in line_up])
me = world.agents[0]
base_n = base_z = 0; deaths = 0; fatal_z = 0; win_z = 0
from collections import Counter
fatal_rows = Counter(); base_rows = Counter()
def dec(i):
    o=[]
    for sz in reversed(cb.FEATURE_SIZES): o.append(i%sz); i//=sz
    return tuple(reversed(o))
for r in range(a.n_rounds):
    world.rng = np.random.default_rng(a.seed + r); np.random.seed(a.seed + r)
    world.new_round(); world.user_input = None
    hist = []
    while world.running:
        if not me.dead:
            idx = cb.state_to_features(world.get_state_for_agent(me))
            hist.append(idx); base_n += 1; base_z += zero[idx]
            base_rows[dec(idx)[4]] += 1
        alive = not me.dead
        world.do_step()
        if alive and me.dead:
            deaths += 1
            fatal_z += zero[hist[-1]]
            fatal_rows[(dec(hist[-1])[4], bool(zero[hist[-1]]))] += 1
            win_z += any(zero[i] for i in hist[-a.window:])
print(f"table: {zero.sum()}/{len(zero)} rows all-zero ({100*zero.mean():.1f} %)")
print(f"field {a.n_opponents}x{a.opponents if a.n_opponents else 'none'}, {a.n_rounds} rounds")
print(f"  base rate over all steps      : {100*base_z/base_n:6.2f} %  ({base_z}/{base_n})")
print(f"  at the fatal step             : {100*fatal_z/max(deaths,1):6.2f} %  ({fatal_z}/{deaths} deaths)")
print(f"  anywhere in the last {a.window} steps  : {100*win_z/max(deaths,1):6.2f} %")
print("  digit5 (own danger) at fatal step, (danger, all-zero-row): ", dict(sorted(fatal_rows.items())))
print("  digit5 over all steps: ", {k: round(100*v/base_n,1) for k,v in sorted(base_rows.items())})

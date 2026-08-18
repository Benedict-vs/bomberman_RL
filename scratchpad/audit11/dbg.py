import os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))
import settings as s
from environment import BombeRLeWorld, WorldArgs
import agent_code.benedict_task4.callbacks as cb
from hunt_ceiling_v2 import HuntCeiling, dist_map, target_cells
SAFE = cb.SAFE
q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = HuntCeiling(q, 4, simultaneous=True)
log_dir = REPO / "logs" / "audit11"; log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                 match_name="a11dbg", seed=990731, silence_errors=False, scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None
n_cells = [0, 0]   # total cells before / after the danger>0 filter
hist = {}
for r in range(10):
    world.rng = np.random.default_rng(990731 + r); np.random.seed(990731 + r)
    world.new_round()
    while world.running:
        alive = [a for a in world.active_agents if a.code_name == "user_agent"]
        action = "WAIT"
        if alive:
            gs = world.get_state_for_agent(alive[0])
            if gs is not None:
                danger = cb.danger_map(gs); field = gs["field"]
                bombs = {p for p, _ in gs["bombs"]}
                for o in gs["others"]:
                    p = o[3]
                    cells = target_cells(p, field, bombs)
                    kept = [c for c in cells if c == p or danger[c] > 0]
                    n_cells[0] += len(cells); n_cells[1] += len(kept)
                    hist[(len(cells), len(kept))] = hist.get((len(cells), len(kept)), 0) + 1
                action = policy.act(gs)
        world.do_step(action)
world.end()
print("cells before filter:", n_cells[0], " after:", n_cells[1])
print("(len_before, len_after) histogram:", dict(sorted(hist.items())))

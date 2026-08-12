"""When is digit 6 = NO_TARGET, and why? Dump a few real states."""
from __future__ import annotations
import importlib, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environment import BombeRLeWorld, WorldArgs   # noqa: E402

agent = "benedict_task3"
cb = importlib.import_module(f"agent_code.{agent}.callbacks")
q = np.load(cb.MODEL_FILE)
log_dir = ROOT / "logs" / "audit_nt"; log_dir.mkdir(parents=True, exist_ok=True)

def run(opponents, n_opp, n_rounds, seed=550731, dump=0):
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit_nt", seed=None,
                      silence_errors=False, scenario="classic")
    line_up = [agent] + [opponents] * n_opp if n_opp else [agent]
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])
    me = world.agents[0]
    steps = 0; nt = 0; dumped = 0; crates_left = []
    for r in range(n_rounds):
        world.rng = np.random.default_rng(seed + r); np.random.seed(seed + r)
        world.new_round(); world.user_input = None
        while world.running:
            if not me.dead:
                gs = world.get_state_for_agent(me)
                x, y = gs["self"][3]
                d = cb.danger_map(gs)
                own = 0 if d[x, y] >= cb.SAFE else int(d[x, y]) + 1
                steps += 1
                if own == 0:
                    tgt, dist = cb.target_direction(x, y, gs["field"], gs["coins"])
                    if tgt == cb.NO_TARGET:
                        nt += 1
                        crates_left.append(int((gs["field"] == 1).sum()))
                        if dumped < dump:
                            dumped += 1
                            print(f"\n--- round {r} step {gs['step']} at {(x,y)}  "
                                  f"crates on board {(gs['field']==1).sum()}  "
                                  f"coins visible {len(gs['coins'])}  "
                                  f"others {[o[3] for o in gs['others']]}")
                            f = gs["field"]
                            for yy in range(max(0,y-4), min(f.shape[1], y+5)):
                                print("   " + "".join(
                                    ("@" if (xx,yy)==(x,y) else
                                     "#" if f[xx,yy]==-1 else "C" if f[xx,yy]==1 else ".")
                                    for xx in range(max(0,x-6), min(f.shape[0], x+7))))
            world.do_step()
    lab = f"{n_opp}x{opponents}" if n_opp else "solo"
    cl = np.array(crates_left) if crates_left else np.array([0])
    print(f"{lab:<26} NO_TARGET on {100*nt/steps:5.2f} % of safe steps "
          f"({nt}/{steps});  crates still on board at those steps: "
          f"median {np.median(cl):.0f}, max {cl.max()}")

run("rule_based_agent", 3, 30)
run("peaceful_agent", 3, 30)

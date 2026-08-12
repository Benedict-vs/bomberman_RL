"""Where do the ~24 INVALID_ACTIONs per round in company come from?

Reproduces act()'s choice from the table, then classifies the step: was the
chosen action invalid, was the row all-zero, and what blocked it -- a wall/crate
(static, visible to BFS) or a body/bomb (dynamic, invisible to target_direction)?
"""
from __future__ import annotations
import argparse, importlib, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environment import BombeRLeWorld, WorldArgs   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="benedict_task3")
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()

    cb = importlib.import_module(f"agent_code.{a.agent}.callbacks")
    q = np.load(cb.MODEL_FILE)
    log_dir = ROOT / "logs" / "audit_inv"; log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit_inv", seed=None,
                      silence_errors=False, scenario="classic")
    line_up = [a.agent] + [a.opponents] * a.n_opponents if a.n_opponents else [a.agent]
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])
    me = world.agents[0]
    rng = np.random.default_rng(cb.POLICY_SEED)

    c = Counter(); steps = 0
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r); np.random.seed(a.seed + r)
        world.new_round(); world.user_input = None
        while world.running:
            if not me.dead:
                gs = world.get_state_for_agent(me)
                row = cb.state_to_features(gs)
                qr = q[row]
                best = np.flatnonzero(qr >= qr.max() - cb.TIE_TOL)
                ai = int(best[0] if best.size == 1 else rng.choice(best))
                act = cb.ACTIONS[ai]
                zero = not np.abs(qr).sum()
                steps += 1
                c["zero_row"] += zero
                x, y = gs["self"][3]
                field = gs["field"]
                occupied = {p for p, _ in gs["bombs"]}
                occupied.update(o[3] for o in gs["others"])
                invalid = False; why = ""
                if ai < 4:
                    dx, dy = cb.DELTAS[ai]
                    nx, ny = x + dx, y + dy
                    if field[nx, ny] != 0:
                        invalid, why = True, "wall_or_crate"
                    elif (nx, ny) in occupied:
                        invalid, why = True, ("bomb" if (nx, ny) in {p for p, _ in gs["bombs"]} else "body")
                elif act == "BOMB" and not gs["self"][2]:
                    invalid, why = True, "no_bomb_left"
                if invalid:
                    c["invalid"] += 1
                    c["inv_" + why] += 1
                    c["inv_zero" if zero else "inv_valued"] += 1
                    # did digit 6 point at the blocked tile?
                    feats = decode(cb, row)
                    if ai < 4 and feats[5] - 1 == ai:
                        c["inv_target_dir"] += 1
                c["act_" + act] += 1
            world.do_step()

    n = a.n_rounds
    print(f"{steps/n:.1f} steps/ep, {c['invalid']/n:.2f} invalid/ep "
          f"({100*c['invalid']/steps:.1f} % of steps); all-zero rows "
          f"{100*c['zero_row']/steps:.2f} % of steps")
    for k in sorted(c):
        if k.startswith(("inv_", "act_")):
            print(f"  {k:<20}{c[k]/n:8.2f}/ep  {100*c[k]/max(c['invalid'],1) if k.startswith('inv_') else 100*c[k]/steps:6.1f} %")


def decode(cb, idx):
    out = []
    for size in reversed(cb.FEATURE_SIZES):
        out.append(idx % size); idx //= size
    return list(reversed(out))


if __name__ == "__main__":
    main()

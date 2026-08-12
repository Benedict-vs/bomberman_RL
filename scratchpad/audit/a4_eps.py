"""Roll out a fixed table at a chosen epsilon in a chosen field. No learning.

Tests E25's claim that "the warm-started table meets eps=0.2 in a field that
kills it in 40 steps, every single round", and separates the field from epsilon.
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environment import BombeRLeWorld, WorldArgs   # noqa: E402
import events as e                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="benedict_task3")
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--n-rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=810731)
    a = ap.parse_args()

    import importlib
    cb = importlib.import_module(f"agent_code.{a.agent}.callbacks")
    orig_act = cb.act
    rng = np.random.default_rng(20260731)

    def eps_act(self, gs):
        if rng.random() < a.eps:
            return cb.ACTIONS[int(rng.integers(len(cb.ACTIONS)))]
        return orig_act(self, gs)
    cb.act = eps_act

    log_dir = ROOT / "logs" / "audit_eps"; log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit_eps", seed=None,
                      silence_errors=False, scenario="classic")
    line_up = [a.agent] + [a.opponents] * a.n_opponents if a.n_opponents else [a.agent]
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])
    me = world.agents[0]

    ev = Counter(); steps = 0
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)
        world.new_round(); world.user_input = None
        while world.running:
            alive = not me.dead
            world.do_step()
            if alive:
                for v in me.events: ev[v] += 1
                steps += 1
    n = a.n_rounds
    print(f"eps={a.eps:<5} field={a.n_opponents}x{a.opponents if a.n_opponents else 'none':<22} "
          f"steps/ep {steps/n:6.1f}  suicide {ev[e.KILLED_SELF]/n:5.3f}  "
          f"killed_by {(ev[e.GOT_KILLED]-ev[e.KILLED_SELF])/n:5.3f}  "
          f"survived {1-ev[e.GOT_KILLED]/n:5.3f}  crates {ev[e.CRATE_DESTROYED]/n:6.2f}  "
          f"bombs {ev[e.BOMB_DROPPED]/n:5.2f}  invalid {ev[e.INVALID_ACTION]/n:6.2f}  "
          f"coins {ev[e.COIN_COLLECTED]/n:5.2f}")


if __name__ == "__main__":
    main()

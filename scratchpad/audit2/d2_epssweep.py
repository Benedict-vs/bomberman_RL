"""Evaluate a frozen table at a chosen epsilon / tie tolerance.

Same arena-matching as tools/evaluate.py (world.rng and np.random reseeded to
base_seed + round_index before every round), so numbers are comparable to
results/eval/task3_opponents/.  The only difference is that `act` is wrapped so
the greedy policy can be softened -- which is the thing under test.

One config per process: callbacks.py reads BM_* at import time.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_rung2ship")   # BM_MODEL_SUFFIX
    ap.add_argument("--hunt", default="1")
    ap.add_argument("--tie-tol", default="0.0")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=550731)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.environ["BM_MODEL_SUFFIX"] = a.suffix
    os.environ["BM_HUNT"] = a.hunt
    os.environ["BM_TIE_TOL"] = a.tie_tol
    os.environ["BM_QUIET_LOGS"] = "1"

    import numpy as np
    import importlib
    from environment import BombeRLeWorld, WorldArgs

    cb = importlib.import_module("agent_code.benedict_task3.callbacks")
    orig_act = cb.act
    rng = np.random.default_rng(20260731)

    if a.eps > 0:
        def eps_act(self, gs):
            if rng.random() < a.eps:
                return cb.ACTIONS[int(rng.integers(len(cb.ACTIONS)))]
            return orig_act(self, gs)
        cb.act = eps_act

    log_dir = ROOT / "logs" / "audit2"; log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit2", seed=a.seed,
                      silence_errors=False, scenario="classic")
    line_up = ["benedict_task3"] + [a.opponents] * a.n_opponents
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])
    me = world.agents[0]

    keys = ["score", "coins", "kills", "suicides", "crates", "bombs", "invalid", "steps"]
    acc = {k: [] for k in keys}
    acc["survived"] = []
    acc["won"] = []
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)
        world.new_round(); world.user_input = None
        while world.running:
            world.do_step()
        for k in keys:
            acc[k].append(me.statistics.get(k, 0))
        acc["survived"].append(int(not me.dead))
        best = max(ag.score for ag in world.agents)
        acc["won"].append(int(me.score >= best and sum(ag.score == best for ag in world.agents) == 1))

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    tag = a.tag or f"{a.suffix} hunt={a.hunt} tol={a.tie_tol} eps={a.eps}"
    print(f"{tag:<46} score {m['score']:7.3f}  coins {m['coins']:6.3f}  kills {m['kills']:5.3f}  "
          f"crates {m['crates']:7.3f}  bombs {m['bombs']:6.2f}  inval {m['invalid']:7.2f}  "
          f"suic {m['suicides']:5.3f}  surv {m['survived']:5.3f}  won {m['won']:5.3f}  "
          f"steps {m['steps']:6.1f}", flush=True)


if __name__ == "__main__":
    main()

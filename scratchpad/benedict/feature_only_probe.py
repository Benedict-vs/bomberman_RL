"""Does the learned table still matter, or do the features now decide the game?

E26's shipping candidate is the *frozen rung-2 table* read through the rung-3
feature map. That invites the obvious objection, and `AGENTS.md`'s hardest rule
is exactly this one: the model must **learn from** the features, and a feature
that returns "the best action" is not allowed.

So this builds the strongest purely-feature policy the same digits permit --
escape if in danger, bomb if the bomb digit says it is useful, otherwise walk the
direction digit 6 points -- and plays it in the same arenas. It is deliberately a
rule-based agent, used as a *measuring stick*, never as a submission.

  policy_greedy  >= learned   ->  the table is decoration; the features are the
                                  agent, and E26 cannot ship as machine learning
  policy_greedy  <  learned   ->  the table contributes, and the gap is the size
                                  of that contribution

Run:  BM_HUNT=1 uv run python scratchpad/benedict/feature_only_probe.py --n-rounds 300
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import events as e                                   # noqa: E402
from environment import BombeRLeWorld, WorldArgs     # noqa: E402


def load_callbacks(agent: str):
    """The *same* module object the framework imports, so patching `act` works.

    Loading it by file path instead gives a second, independent module: the
    monkeypatch would then apply to a copy nobody plays with, and both arms of
    this probe would silently run the learned policy.
    """
    import importlib
    return importlib.import_module(f"agent_code.{agent}.callbacks")


def greedy_from_features(cb, gs) -> str:
    """The best policy the digits allow, with no value function at all."""
    field = gs["field"]
    x, y = gs["self"][3]
    have_bomb = gs["self"][2]
    danger = cb.danger_map(gs)
    occupied = {p for p, _ in gs["bombs"]}
    occupied.update(o[3] for o in gs["others"])
    others = [o[3] for o in gs["others"]]

    own_danger = 0 if danger[x, y] >= cb.SAFE else int(danger[x, y]) + 1
    if own_danger:
        step = cb.escape_direction(x, y, field, danger, occupied)
        return cb.ACTIONS[step - 1] if step != cb.NO_TARGET else "WAIT"
    if have_bomb and cb.bomb_hits_crate(x, y, field, others):
        return "BOMB"
    step, _ = cb.target_direction(x, y, field, gs["coins"], others)
    return cb.ACTIONS[step - 1] if step != cb.NO_TARGET else "WAIT"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="benedict_task3")
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()

    cb = load_callbacks(a.agent)
    print(f"table: {Path(cb.MODEL_FILE).name}   HUNT={cb.HUNT}")
    orig_act = cb.act

    log_dir = ROOT / "logs" / "feature_probe"
    log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="feature_probe", seed=None,
                      silence_errors=True, scenario="classic")

    for mode in ("learned", "features-only"):
        cb.act = orig_act if mode == "learned" else (
            lambda self, gs: greedy_from_features(cb, gs))

        world = BombeRLeWorld(wargs, [(a.agent, False)]
                              + [(a.opponents, False)] * 3)
        me = world.agents[0]
        tot = {"score": 0.0, "kills": 0, "coins": 0, "crates": 0,
               "invalid": 0, "survived": 0}
        for r in range(a.n_rounds):
            world.rng = np.random.default_rng(a.seed + r)
            np.random.seed(a.seed + r)          # same opponents for both modes
            world.new_round()
            while world.running:
                world.do_step()
            st = me.statistics
            for k in ("kills", "coins", "crates", "invalid"):
                tot[k] += st.get(k, 0)
            tot["score"] += me.score
            tot["survived"] += int(not me.dead)
        n = a.n_rounds
        print(f"  {mode:<14} score {tot['score']/n:6.3f}  kills {tot['kills']/n:5.3f}  "
              f"coins {tot['coins']/n:5.3f}  crates {tot['crates']/n:6.2f}  "
              f"invalid {tot['invalid']/n:6.2f}  survived {tot['survived']/n:5.3f}")

    cb.act = orig_act


if __name__ == "__main__":
    main()

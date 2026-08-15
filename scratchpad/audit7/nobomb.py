#!/usr/bin/env python3
"""Audit 7: which decision does opponent proximity change -- escape, or BOMB?

92 % of the shipped agent's deaths are its own bomb (died 0.747, suicides
0.690), and 93.3 % of all deaths happen with an opponent within BFS distance 2
against a 15.3 % base rate. Two readings:

  (a) the escape step is wrong there   -> tested by `near` in condrule.py, which
      FAILED (score -0.210): forcing digit 6 when an opponent is close does not
      help, so the escape action is not the missing answer;
  (b) the BOMB was wrong there         -> tested here. Suppress BOMB whenever an
      opponent is within BFS distance `--near-max`, taking the table's own
      second-best action instead. Placebo arm suppresses BOMB on an equal-sized
      random subset of steps, so "conditioned on the opponent" and "bombs less"
      stay separable.

Evaluation only; no training, no agent file touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from environment import BombeRLeWorld, WorldArgs  # noqa: E402

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
BOMB = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["nb_near", "nb_rand", "nb_far"], required=True)
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=990731)
    ap.add_argument("--near-max", type=int, default=2)
    ap.add_argument("--p-rand", type=float, default=0.153,
                    help="fire probability for the placebo arm = base rate of "
                         "'opponent within 2' over all steps (measured)")
    args = ap.parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    wargs = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=str(log_dir), save_stats=False,
        match_name=f"a7nb_{args.arm}", seed=args.seed, silence_errors=False,
        scenario="classic",
    )
    world = BombeRLeWorld(wargs, [(n, False) for n in
                                  ["benedict_task4"] + ["rule_based_agent"] * 3])

    import agent_code.benedict_task4.callbacks as cb

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    rng = np.random.default_rng(4242)
    fired = {"n": 0, "steps": 0, "bomb_steps": 0}

    def act(self, game_state: dict) -> str:
        row = cb.state_to_features(game_state)
        fired["steps"] += 1
        qr = q[row]
        best = np.flatnonzero(qr >= qr.max())
        choice = int(best[0] if best.size == 1 else self.policy_rng.choice(best))

        if choice == BOMB:
            fired["bomb_steps"] += 1
            if args.arm == "nb_rand":
                hit = rng.random() < args.p_rand
            else:
                field = game_state["field"]
                x, y = game_state["self"][3]
                others = [o[3] for o in game_state["others"]]
                if others:
                    oset = set(others)
                    _, od = cb.bfs_first_step(x, y, field, lambda p: p in oset)
                    if od == 0:
                        od = 99
                else:
                    od = 99
                hit = (od <= args.near_max) if args.arm == "nb_near" else (od > args.near_max)
            if hit:
                fired["n"] += 1
                alt = qr.copy()
                alt[BOMB] = -np.inf
                choice = int(np.argmax(alt))
        return ACTIONS[choice]

    cb.act = act

    rows = []
    try:
        for ri in range(args.n_rounds):
            world.rng = np.random.default_rng(args.seed + ri)
            np.random.seed(args.seed + ri)
            world.new_round()
            while world.running:
                world.do_step()
            me = world.agents[0]
            best_s = max(a.statistics.get("score", 0) for a in world.agents)
            rows.append(dict(
                round=ri, score=me.statistics.get("score", 0),
                won=int(me.statistics.get("score", 0) == best_s),
                kills=me.statistics.get("kills", 0),
                coins=me.statistics.get("coins", 0),
                crates=me.statistics.get("crates", 0),
                bombs=me.statistics.get("bombs", 0),
                suicides=me.statistics.get("suicides", 0),
                survived=int(not me.dead), died=int(me.dead)))
            if (ri + 1) % 50 == 0:
                print(f"  {args.arm} {ri+1}/{args.n_rounds}", file=sys.stderr, flush=True)
    finally:
        world.end()

    out = Path(__file__).resolve().parent / f"nb_{args.arm}.json"
    out.write_text(json.dumps({"arm": args.arm, "seed": args.seed, "n": args.n_rounds,
                               "suppressed": fired["n"], "bomb_steps": fired["bomb_steps"],
                               "steps": fired["steps"], "rows": rows}, indent=0))
    print(f"{args.arm}: score {np.mean([r['score'] for r in rows]):.3f} "
          f"won {np.mean([r['won'] for r in rows]):.3f} "
          f"suppressed {fired['n']}/{fired['bomb_steps']} bomb decisions")


if __name__ == "__main__":
    main()

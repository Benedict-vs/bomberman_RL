#!/usr/bin/env python3
"""Audit 7 probe: what information is missing from the SHIPPED rung-4 table?

Runs the shipped `benedict_task4` agent against 3 x rule_based_agent through the
same world construction `tools/evaluate.py` uses (paired arenas, opponents
seeded per round), and instruments `state_to_features` in-process to record, per
step:

  * the row index and the eight digits
  * whether that row is all-zero in the shipped table (E28's coverage metric)
  * the action gap of the greedy decision
  * digit 7 decomposed: crate in blast / opponent in blast / both
  * BFS distance to the nearest opponent (E28's top-ranked candidate digit)
  * whether the agent died within the next k steps

Nothing here is a fix; it is measurement only. No file outside
scratchpad/audit7/ is written.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import settings as s  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=990731)
    ap.add_argument("--out", type=str, default="probe_ship.pkl")
    args_cli = ap.parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    args = WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=str(log_dir),
        save_stats=False,
        match_name="a7probe",
        seed=args_cli.seed,
        silence_errors=False,
        scenario="classic",
    )

    line_up = ["benedict_task4"] + ["rule_based_agent"] * 3
    world = BombeRLeWorld(args, [(name, False) for name in line_up])

    import agent_code.benedict_task4.callbacks as cb

    q = np.load(Path(REPO_ROOT) / "agent_code/benedict_task4/q_table.npy")
    zero_rows = ~np.any(q != 0.0, axis=1)

    records: list[dict] = []
    state = {"round": -1, "t": 0}

    orig_stf = cb.state_to_features

    def instrumented(game_state: dict) -> int:
        row = orig_stf(game_state)

        field = game_state["field"]
        x, y = game_state["self"][3]
        others = [o[3] for o in game_state["others"]]

        # digit-7 decomposition: what actually made bomb_useful fire (or not)
        blast = cb.blast_coords(x, y, field)
        crate_in_blast = any(field[cx, cy] == 1 for cx, cy in blast)
        opp_in_blast = any(p in blast for p in others)
        have_bomb = bool(game_state["self"][2])

        # candidate digit: BFS distance to the nearest opponent
        if others:
            other_set = set(others)
            _, opp_dist = cb.bfs_first_step(
                x, y, field, lambda pos: pos in other_set
            )
        else:
            opp_dist = -1

        digits = []
        r = row
        for size in reversed(cb.FEATURE_SIZES):
            digits.append(r % size)
            r //= size
        digits = tuple(reversed(digits))

        qrow = q[row]
        order = np.sort(qrow)
        gap = float(order[-1] - order[-2])

        records.append(
            dict(
                round=state["round"],
                t=state["t"],
                row=int(row),
                digits=digits,
                zero=bool(zero_rows[row]),
                gap=gap,
                crate_in_blast=crate_in_blast,
                opp_in_blast=opp_in_blast,
                have_bomb=have_bomb,
                opp_dist=int(opp_dist),
                n_others=len(others),
                greedy=int(np.argmax(qrow)),
            )
        )
        return row

    cb.state_to_features = instrumented

    rounds: list[dict] = []
    try:
        for ri in range(args_cli.n_rounds):
            world.rng = np.random.default_rng(args_cli.seed + ri)
            np.random.seed(args_cli.seed + ri)
            state["round"] = ri
            world.new_round()
            n_before = len(records)
            while world.running:
                state["t"] = len(records) - n_before
                world.do_step()

            me = world.agents[0]
            rounds.append(
                dict(
                    round=ri,
                    died=int(me.dead),
                    suicides=me.statistics.get("suicides", 0),
                    score=me.statistics.get("score", 0),
                    kills=me.statistics.get("kills", 0),
                    crates=me.statistics.get("crates", 0),
                    steps=me.statistics.get("steps", 0),
                    n_steps_logged=len(records) - n_before,
                    best=max(a.statistics.get("score", 0) for a in world.agents),
                )
            )
            if (ri + 1) % 25 == 0:
                print(f"  round {ri + 1}/{args_cli.n_rounds}", file=sys.stderr, flush=True)
    finally:
        world.end()

    out = Path(__file__).resolve().parent / args_cli.out
    with out.open("wb") as fh:
        pickle.dump({"records": records, "rounds": rounds,
                     "seed": args_cli.seed, "n_rounds": args_cli.n_rounds}, fh)
    print(f"wrote {out}  steps={len(records)}  rounds={len(rounds)}")


if __name__ == "__main__":
    main()

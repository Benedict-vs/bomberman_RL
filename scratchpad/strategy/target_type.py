#!/usr/bin/env python3
"""Costing K's best-evidenced feature: target *type* alongside target direction.

Digit 6 gives the first BFS step to "whatever we are after"; digit 8 gives how
far it is. Neither says *what* it is, so the table cannot price walking toward a
coin (+1 on arrival) against a crate (bomb, wait, walk back) against an opponent.
K's only clean drop-two-bits ablation removed exactly this and the agent fell
into period-2 loops.

This measures the joint distribution of (type, distance bucket) over safe steps,
which is what a re-partition of digit 8 would have to fit into five slots.
"""
from __future__ import annotations

import argparse, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
US = "benedict_task4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()
    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame              # noqa: F401
    import agent_code.benedict_task4.callbacks as cb

    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "ttype", "seed": a.seed,
        "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
    world.user_input = None

    joint = Counter()
    safe = 0
    for _ in range(a.rounds):
        world.new_round()
        while world.running and world.step < 400:
            for ag in world.active_agents:
                if ag.code_name != US:
                    continue
                gs = world.get_state_for_agent(ag)
                if gs is None:
                    continue
                x, y = gs["self"][3]
                field = gs["field"]
                danger = cb.danger_map(gs)
                if danger[x, y] < cb.SAFE:
                    continue                      # danger branch: digit 8 is the lattice
                safe += 1
                others = [o[3] for o in gs["others"]]
                coins = set(gs["coins"])
                # replicate target_direction's priority to recover the *type*
                kind = "none"
                dist = 0
                if (x, y) in coins:
                    kind = "onit"
                else:
                    if coins:
                        stp, d = cb.bfs_first_step(x, y, field, lambda p: p in coins)
                        if stp != cb.NO_TARGET:
                            kind, dist = "coin", d
                    if kind == "none":
                        stp, d = cb.bfs_first_step(x, y, field, lambda p: field[p] == 1)
                        if stp != cb.NO_TARGET:
                            kind, dist = "crate", d
                        elif others:
                            oset = set(others)
                            stp, d = cb.bfs_first_step(x, y, field, lambda p: p in oset)
                            if stp != cb.NO_TARGET:
                                kind, dist = "opp", d
                joint[(kind, cb.distance_bucket(dist))] += 1
            world.do_step("WAIT")
    world.end()

    print(f"safe steps {safe} over {a.rounds} rounds")
    print(f"{'type':<8}{'bucket':>8}{'share':>9}   (bucket: 0 none, 1 d=1, 2 d=2, 3 d=3-4, 4 d>=5)")
    for (k, b), n in sorted(joint.items(), key=lambda kv: -kv[1]):
        print(f"{k:<8}{b:>8}{n/safe:>9.4f}")
    by_kind = Counter()
    for (k, b), n in joint.items():
        by_kind[k] += n
    print("\nby type: " + "  ".join(f"{k}:{n/safe:.4f}" for k, n in by_kind.most_common()))
    print("digit 8 currently spends its five slots on the distance bucket alone.")


if __name__ == "__main__":
    main()

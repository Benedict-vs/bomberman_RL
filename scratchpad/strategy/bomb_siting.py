#!/usr/bin/env python3
"""How much is left on the table in bomb *siting*?

E37 bought +0.130 crates/bomb and E38 lost -0.194 over the training horizon, so
this is the quantity the rung is sensitive to. Digit 7 is one bit -- "something
worth destroying is in range" -- so the table cannot distinguish a bomb that
clears one crate from one that clears four, and cannot learn to take one more
step for a better spot.

This records, per armed step, how many crates the hypothetical bomb here would
destroy, how many the best tile within 1-3 steps would destroy, and what the
policy actually did. It bounds what a crate *count* digit (M's feature) could buy
before anyone trains anything.

    uv run python scratchpad/strategy/bomb_siting.py --rounds 60
"""
from __future__ import annotations

import argparse, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
US = "benedict_task4"
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def dist_map(sx, sy, field, blocked):
    d = np.full(field.shape, -1, dtype=np.int16)
    d[sx, sy] = 0; q = [(sx, sy)]; head = 0
    while head < len(q):
        cx, cy = q[head]; head += 1
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if d[nx, ny] != -1 or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            d[nx, ny] = d[cx, cy] + 1; q.append((nx, ny))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()
    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame                  # noqa: F401
    import agent_code.benedict_task4.callbacks as cb

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "siting", "seed": a.seed,
        "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
    world.user_input = None

    here = Counter()                 # crates a bomb here would clear
    bombed_here = Counter()          # ... when the policy actually chose BOMB
    best_within = defaultdict(list)  # radius -> best count reachable
    gain = defaultdict(list)
    armed = 0

    for _ in range(a.rounds):
        world.new_round()
        while world.running and world.step < 400:
            for ag in world.active_agents:
                if ag.code_name != US:
                    continue
                gs = world.get_state_for_agent(ag)
                if gs is None or not gs["self"][2]:
                    continue
                armed += 1
                x, y = gs["self"][3]
                field = gs["field"]
                bombs = {p for p, _ in gs["bombs"]}
                n_here = sum(1 for c in cb.blast_coords(x, y, field) if field[c] == 1)
                here[n_here] += 1
                act = cb.ACTIONS[int(np.argmax(q[cb.state_to_features(gs)]))]
                if act == "BOMB":
                    bombed_here[n_here] += 1
                dm = dist_map(x, y, field, bombs)
                # crate yield of every free tile, computed once per step
                yields = {}
                for bx in range(field.shape[0]):
                    for by in range(field.shape[1]):
                        if 0 <= dm[bx, by] <= 3:
                            yields[(bx, by)] = sum(
                                1 for c2 in cb.blast_coords(bx, by, field)
                                if field[c2] == 1)
                for radius in (1, 2, 3):
                    best = max([n_here] + [v for t, v in yields.items()
                                           if dm[t] <= radius])
                    best_within[radius].append(best)
                    gain[radius].append(best - n_here)
            world.do_step("WAIT")
    world.end()

    print(f"armed steps {armed} over {a.rounds} rounds")
    print("\ncrates a bomb *here* would clear -- distribution over armed steps,")
    print("and the share of those steps where the greedy policy chose BOMB:")
    print(f"{'crates':>7}{'share':>9}{'P(BOMB)':>10}")
    for k in sorted(here):
        print(f"{k:>7}{here[k]/armed:>9.3f}{bombed_here[k]/here[k]:>10.3f}")
    exp_here = sum(k * n for k, n in here.items()) / armed
    exp_bomb = (sum(k * n for k, n in bombed_here.items())
                / max(sum(bombed_here.values()), 1))
    print(f"\nmean crates in range while armed        {exp_here:.3f}")
    print(f"mean crates cleared by the bombs taken   {exp_bomb:.3f}   "
          f"(policy bombs on {sum(bombed_here.values())/armed:.1%} of armed steps)")
    for radius in (1, 2, 3):
        print(f"best tile within {radius} step(s): mean {np.mean(best_within[radius]):.3f} "
              f"crates, i.e. +{np.mean(gain[radius]):.3f} over standing still; "
              f"a better tile exists on {np.mean(np.array(gain[radius])>0):.1%} of armed steps")


if __name__ == "__main__":
    main()

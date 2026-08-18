#!/usr/bin/env python3
"""When is the score actually earned? A time profile of one `classic` round.

The eval CSVs are per-round totals, so they cannot say whether the second half
of a round is worth anything. This instruments the world directly and records,
per step: coins still on the board / not yet revealed, crates remaining, agents
alive, and each agent's running score. 9 coins exist in total (settings.py:28).

If the coin economy closes early, "survive longer" cannot be a scoring lever
after that point, and the only currency left is kills. That is the mechanism the
rung-4 ledger leaves open when it says six survival interventions did not
convert into points.

    uv run python scratchpad/strategy/round_economy.py --rounds 150
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
US = "benedict_task4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()
    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame            # noqa: F401

    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "round_economy",
        "seed": a.seed, "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
    world.user_input = None

    T = 401
    our_coins = np.zeros(T)      # cumulative, averaged over rounds
    opp_coins = np.zeros(T)
    our_kills = np.zeros(T)
    opp_kills = np.zeros(T)
    crates = np.zeros(T)
    on_board = np.zeros(T)       # coins visible and uncollected
    alive_us = np.zeros(T)
    alive_opp = np.zeros(T)
    seen = np.zeros(T)
    our_death_step, opp_death_steps = [], []

    for _ in range(a.rounds):
        world.new_round()
        prev_alive = {ag.name for ag in world.active_agents}
        t = 0
        while world.running and world.step < 400:
            world.do_step("WAIT")
            t = world.step
            if t >= T:
                break
            seen[t] += 1
            for ag in world.agents:
                st = ag.statistics
                if ag.code_name == US:
                    our_coins[t] += st.get("coins", 0)
                    our_kills[t] += st.get("kills", 0)
                else:
                    opp_coins[t] += st.get("coins", 0)
                    opp_kills[t] += st.get("kills", 0)
            crates[t] += int((world.arena == 1).sum())
            on_board[t] += sum(1 for c in world.coins if c.collectable)
            now = {ag.name for ag in world.active_agents}
            for gone in prev_alive - now:
                (our_death_step if gone.startswith("benedict") else opp_death_steps).append(t)
            prev_alive = now
            alive_us[t] += int(any(ag.code_name == US for ag in world.active_agents))
            alive_opp[t] += sum(1 for ag in world.active_agents if ag.code_name != US)
    world.end()

    print(f"rounds {a.rounds}")
    print(f"{'step':>5} {'rounds':>6} {'ourCoin':>8} {'oppCoin':>8} {'left':>6} "
          f"{'crates':>7} {'ourKill':>8} {'oppKill':>8} {'usAlive':>8} {'oppAlive':>8}")
    for t in list(range(20, 401, 20)):
        if seen[t] == 0:
            continue
        n = seen[t]
        print(f"{t:>5} {int(n):>6} {our_coins[t]/n:>8.3f} {opp_coins[t]/n:>8.3f} "
              f"{on_board[t]/n:>6.2f} {crates[t]/n:>7.1f} {our_kills[t]/n:>8.3f} "
              f"{opp_kills[t]/n:>8.3f} {alive_us[t]/n:>8.3f} {alive_opp[t]/n:>8.3f}")

    ok = np.array(our_death_step)
    od = np.array(opp_death_steps)
    print(f"\nour deaths n={len(ok)} of {a.rounds} rounds; step "
          f"median {np.median(ok) if len(ok) else float('nan'):.0f} "
          f"quartiles {np.percentile(ok,[25,75]) if len(ok) else ''}")
    print(f"opponent deaths n={len(od)}; step median "
          f"{np.median(od) if len(od) else float('nan'):.0f} "
          f"quartiles {np.percentile(od,[25,75]) if len(od) else ''}")
    for lo, hi in ((0,100),(100,200),(200,300),(300,401)):
        print(f"  deaths in [{lo},{hi}): ours {int(((ok>=lo)&(ok<hi)).sum())}, "
              f"opponents {int(((od>=lo)&(od<hi)).sum())}")


if __name__ == "__main__":
    main()

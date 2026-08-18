#!/usr/bin/env python3
"""Is a kill easier while crates are still standing? And can we get there in time?

Benedict's reading from watching play: the agent survives to the empty board, and
an empty board is the *worst* place to kill because everything is outrunnable --
so the hunt should happen while crates still provide walls to pin somebody
against, i.e. during the crate phase, by working toward an opponent instead of
treating it as the last-resort objective (which digit 6 only does when no coin
and no crate is reachable, so today: never before the board is stripped).

`trap_opportunity.py` counted only bombs at the tile we already stand on. This
counts *constructible* traps: for each opponent, every free tile whose blast
reaches it and from which it would have no survivable escape -- whether or not we
are standing there -- plus how far we would have to walk to reach the nearest
such tile. That is the supply side of the proposed strategy.

Also checks the premise: are we actually alive when the board runs out?

Opponent movement is modelled as free (they may cross each other) and they never
drop their own bomb, so "trappable" is an upper bound. They also move while we
walk, which cuts both ways and is not modelled at all.

    uv run python scratchpad/strategy/hunt_window.py --rounds 60 --stride 3
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
    """BFS distance from (sx, sy) over free tiles; -1 where unreachable."""
    d = np.full(field.shape, -1, dtype=np.int16)
    d[sx, sy] = 0
    q = [(sx, sy)]
    head = 0
    while head < len(q):
        cx, cy = q[head]; head += 1
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if d[nx, ny] != -1 or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            d[nx, ny] = d[cx, cy] + 1
            q.append((nx, ny))
    return d


def escapable(sx, sy, field, danger, blocked, horizon):
    if danger[sx, sy] >= horizon:
        return True
    q = [((sx, sy), 0)]; seen = {(sx, sy)}; head = 0
    while head < len(q):
        (cx, cy), depth = q[head]; head += 1
        if depth >= horizon:
            continue
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= horizon:
                return True
            seen.add((nx, ny)); q.append(((nx, ny), depth + 1))
    return False


CRATE_BUCKETS = [(90, 1e9, "90+"), (60, 90, "60-89"), (30, 60, "30-59"),
                 (10, 30, "10-29"), (3, 10, "3-9"), (1, 3, "1-2"), (0, 1, "0")]


def bucket(n):
    for lo, hi, name in CRATE_BUCKETS:
        if lo <= n < hi:
            return name
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()
    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame                     # noqa: F401
    import agent_code.benedict_task4.callbacks as cb
    import settings as s

    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "hunt", "seed": a.seed,
        "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
    world.user_input = None
    SAFE = cb.SAFE

    st = defaultdict(Counter)          # crate bucket -> counters
    dsum = defaultdict(list)           # crate bucket -> distances
    walk = defaultdict(list)           # crate bucket -> walk distance to a trap site
    premise = []                       # per round: (crate-exhaust step, our death step)

    for _ in range(a.rounds):
        world.new_round()
        exhaust = None
        our_death = None
        while world.running and world.step < 400:
            if (world.arena == 1).sum() == 0 and exhaust is None:
                exhaust = world.step
            me = [ag for ag in world.active_agents if ag.code_name == US]
            if not me and our_death is None:
                our_death = world.step
            if me and world.step % a.stride == 0:
                gs = world.get_state_for_agent(me[0])
                if gs is not None:
                    x, y = gs["self"][3]
                    field = gs["field"]
                    base = cb.danger_map(gs)
                    bombs = {p for p, _ in gs["bombs"]}
                    others = [o[3] for o in gs["others"]]
                    ncr = int((field == 1).sum())
                    b = bucket(ncr)
                    st[b]["steps"] += 1
                    if others:
                        dm = dist_map(x, y, field, bombs)
                        od = [dm[p] for p in others if dm[p] >= 0]
                        if od:
                            dsum[b].append(min(od))
                            st[b]["opp_reachable"] += 1
                        # constructible traps against the nearest reachable opponent
                        near = min(((dm[p], p) for p in others if dm[p] >= 0),
                                   default=None)
                        if near is not None:
                            p = near[1]
                            sites = []
                            for bx in range(field.shape[0]):
                                for by in range(field.shape[1]):
                                    if field[bx, by] != 0 or (bx, by) in bombs:
                                        continue
                                    if abs(bx - p[0]) + abs(by - p[1]) > s.BOMB_POWER:
                                        continue
                                    if p not in cb.blast_coords(bx, by, field):
                                        continue
                                    hyp = base.copy()
                                    for (cx2, cy2) in cb.blast_coords(bx, by, field):
                                        hyp[cx2, cy2] = min(hyp[cx2, cy2], s.BOMB_TIMER)
                                    if not escapable(p[0], p[1], field, hyp,
                                                     bombs | {(bx, by)}, SAFE):
                                        sites.append((bx, by))
                            if sites:
                                st[b]["trappable"] += 1
                                reach = [dm[t] for t in sites if dm[t] >= 0]
                                if reach:
                                    st[b]["trap_site_reachable"] += 1
                                    walk[b].append(min(reach))
                                    if min(reach) == 0:
                                        st[b]["trap_site_is_here"] += 1
                                    if min(reach) <= 4:
                                        st[b]["trap_site_within_4"] += 1
            world.do_step("WAIT")
        premise.append((exhaust, our_death if our_death is not None else 400))
    world.end()

    print(f"rounds {a.rounds}, sampling every {a.stride} steps\n")
    print("== premise: are we alive when the board runs out of crates? ==")
    ex = [e for e, _ in premise if e is not None]
    alive_at_ex = [d > e for e, d in premise if e is not None]
    never = sum(1 for e, _ in premise if e is None)
    print(f"  crates fully cleared in {len(ex)}/{a.rounds} rounds, median step "
          f"{np.median(ex) if ex else float('nan'):.0f} "
          f"(quartiles {np.percentile(ex,[25,75]).round(0) if ex else ''})")
    print(f"  we are still alive at that moment in {np.mean(alive_at_ex):.1%} of them")
    print(f"  rounds where crates never fully cleared: {never}")
    dd = [d for _, d in premise]
    print(f"  our death/end step: median {np.median(dd):.0f}, "
          f"died before step 200 in {np.mean([d<200 for d in dd]):.1%} of rounds\n")

    print("== trap supply vs how much of the board is still standing ==")
    print(f"{'crates':>8}{'steps':>7}{'d(opp)':>8}{'trappable':>11}"
          f"{'site reach.':>12}{'walk':>7}{'<=4 away':>10}{'here':>7}")
    for _, _, name in CRATE_BUCKETS:
        c = st.get(name)
        if not c or c["steps"] < 20:
            continue
        n = c["steps"]
        print(f"{name:>8}{n:>7}"
              f"{np.mean(dsum[name]) if dsum[name] else float('nan'):>8.1f}"
              f"{c['trappable']/n:>11.3%}{c['trap_site_reachable']/n:>12.3%}"
              f"{np.mean(walk[name]) if walk[name] else float('nan'):>7.1f}"
              f"{c['trap_site_within_4']/n:>10.3%}{c['trap_site_is_here']/n:>7.3%}")
    print("\n  d(opp)      BFS distance to the nearest reachable opponent")
    print("  trappable   some free tile exists whose bomb leaves it no escape")
    print("  walk        our BFS distance to the nearest such tile")


if __name__ == "__main__":
    main()

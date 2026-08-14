"""Candidate extra feature digits, evaluated on every step the agent was alive.

Each one is a *state* function the agent could compute in `act()` within the
0.5 s budget, and each is the answer to one of the death causes in `analyse.py`.
They are only measured here -- nothing in `agent_code/` is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sim  # noqa: E402


def state_from_gs(gs) -> sim.State:
    """`sim.State` from the dict the agent itself receives."""
    arena = np.array(gs["field"], dtype=np.int8)
    x, y = gs["self"][3]
    agents = {"ME": [int(x), int(y), bool(gs["self"][2]), False]}
    for name, _, left, pos in gs["others"]:
        agents[name] = [int(pos[0]), int(pos[1]), bool(left), False]
    bombs = [[int(p[0]), int(p[1]), int(t), "?"] for p, t in gs["bombs"]]
    # `explosion_map` is `timer - 1` of every stage-0 explosion, which is all the
    # simulator needs; blast shape is irrelevant once it is burning.
    explosions = []
    em = gs["explosion_map"]
    burning = [(int(i), int(j)) for i, j in zip(*np.nonzero(em > 0))]
    if burning:
        explosions.append([burning, 2, 0, "?"])
    st = sim.State(arena, bombs, explosions, agents)
    return st


def bfs_dist(arena, start, goals: set, cap: int = 40) -> int:
    """Walking distance to the nearest goal tile; `cap` if unreachable."""
    if not goals:
        return cap
    seen = {start}
    frontier = [start]
    for depth in range(1, cap + 1):
        nxt = []
        for (x, y) in frontier:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                p = (x + dx, y + dy)
                if p in seen:
                    continue
                if p in goals:
                    return depth
                if arena[p] != 0:
                    continue
                seen.add(p)
                nxt.append(p)
        frontier = nxt
        if not frontier:
            break
    return cap


def bucket(d: int) -> int:
    if d <= 1:
        return 1
    if d == 2:
        return 2
    if d <= 4:
        return 3
    return 4


def compute(gs) -> tuple:
    """The candidate digits, in the order documented in REPORT.md."""
    arena = np.array(gs["field"], dtype=np.int8)
    x, y = gs["self"][3]
    x, y = int(x), int(y)
    have_bomb = bool(gs["self"][2])
    others = [(int(o[3][0]), int(o[3][1])) for o in gs["others"]]
    others_armed = [(int(o[3][0]), int(o[3][1])) for o in gs["others"] if o[2]]

    st = state_from_gs(gs)

    # C1 -- correct, time-aware "is there any way to survive from here"
    c1_escape_timed = int(sim.survivable(st))

    # C2 -- "if I drop a bomb right now, do I still get out?"  Opponents frozen.
    if have_bomb:
        probe = st.clone()
        probe.do_step({"ME": "BOMB"}, ["ME"] + [n for n in probe.agents if n != "ME"])
        c2_bomb_safe = int(sim.survivable(probe))
    else:
        c2_bomb_safe = 0

    # C3 -- how close the nearest opponent is
    c3_opp_dist = bucket(bfs_dist(arena, (x, y), set(others)))

    # C4 -- free neighbours of my own tile: 1 = dead end / corridor end
    c4_exits = sum(1 for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
                   if arena[x + dx, y + dy] == 0)

    # C5 -- would a bomb dropped by an opponent *where it stands now* cover me?
    c5_enemy_threat = 0
    for ox, oy in others_armed:
        if (x, y) in sim.blast_coords(ox, oy, arena):
            c5_enemy_threat = 1
            break

    # C6 -- C5 and I could not get out of that bomb: the trap, one step early
    c6_trap = 0
    if c5_enemy_threat:
        c6_trap = 1
        for ox, oy in others_armed:
            if (x, y) not in sim.blast_coords(ox, oy, arena):
                continue
            probe = st.clone()
            probe.bombs.append([ox, oy, sim.BOMB_TIMER, "OPP"])
            if sim.survivable(probe):
                c6_trap = 0
            else:
                c6_trap = 1
                break

    # C7 -- distance to the nearest tile no live bomb reaches
    lethal = sim.lethal_schedule(st)
    ever = set().union(*lethal) if lethal else set()
    c7_safe_dist = bucket(bfs_dist(arena, (x, y), {p for p in _free_tiles(arena)
                                                   if p not in ever and p != (x, y)}))
    if (x, y) not in ever:
        c7_safe_dist = 0

    # C8 -- how many neighbours digits 1-4 call NB_BLOCKED are an opponent's
    # *body* rather than a wall or a crate. A body moves; a wall does not, and
    # the current map cannot tell the two apart.
    other_set = set(others)
    c8_blocked_by_agent = sum(1 for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
                              if (x + dx, y + dy) in other_set)

    # C9 -- my own bomb is somewhere on the board, so I cannot drop another.
    # Digit 7 folds `bombs_left` in only when a bomb here would also pay off.
    c9_no_bomb = int(not have_bomb)

    return (c1_escape_timed, c2_bomb_safe, c3_opp_dist, c4_exits,
            c5_enemy_threat, c6_trap, c7_safe_dist,
            c8_blocked_by_agent, c9_no_bomb)


_FREE_CACHE: dict = {}


def _free_tiles(arena):
    key = arena.tobytes()
    hit = _FREE_CACHE.get(key)
    if hit is None:
        hit = {(int(i), int(j)) for i, j in zip(*np.nonzero(arena == 0))}
        if len(_FREE_CACHE) > 64:
            _FREE_CACHE.clear()
        _FREE_CACHE[key] = hit
    return hit


NAMES = ["C1_escape_timed", "C2_bomb_safe", "C3_opp_dist", "C4_exits",
         "C5_enemy_threat", "C6_trap", "C7_safe_dist",
         "C8_blocked_by_agent", "C9_no_bomb"]

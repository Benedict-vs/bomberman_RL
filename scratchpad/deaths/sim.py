"""A minimal re-implementation of one `do_step`, for counterfactuals.

Mirrors `environment.py:do_step` -> `poll_and_run_agents` / `update_explosions`
/ `update_bombs` / `evaluate_explosions` and `items.py:Bomb/Explosion`. Coin
collection is omitted: it cannot kill anyone and the only question asked here is
who is alive.

`validate.py` replays each recorded death through this simulator and checks the
death step and the killer against what the real engine produced -- if the timing
model here is wrong, that check fails.
"""

from __future__ import annotations

import numpy as np

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def blast_coords(x: int, y: int, arena) -> list[tuple[int, int]]:
    coords = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if arena[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


class State:
    """Mutable world state: arena, bombs, explosions, agents."""

    __slots__ = ("arena", "bombs", "explosions", "agents", "step")

    def __init__(self, arena, bombs, explosions, agents, step=0):
        self.arena = arena
        self.bombs = bombs            # [ [x, y, timer, owner] ]
        self.explosions = explosions  # [ [coords(list), timer, stage, owner] ]
        self.agents = agents          # {name: [x, y, bombs_left, dead]}
        self.step = step

    def clone(self) -> "State":
        return State(
            self.arena.copy(),
            [b[:] for b in self.bombs],
            [[e[0], e[1], e[2], e[3]] for e in self.explosions],
            {k: v[:] for k, v in self.agents.items()},
            self.step,
        )

    def alive(self, name: str) -> bool:
        return not self.agents[name][3]

    # -- one engine step ---------------------------------------------------
    def tile_free(self, x, y) -> bool:
        if self.arena[x, y] != 0:
            return False
        for b in self.bombs:
            if b[0] == x and b[1] == y:
                return False
        for a in self.agents.values():
            if not a[3] and a[0] == x and a[1] == y:
                return False
        return True

    def do_step(self, actions: dict, order: list[str]) -> None:
        self.step += 1
        for name in order:
            a = self.agents[name]
            if a[3]:
                continue
            act = actions.get(name)
            if act in DELTAS:
                dx, dy = DELTAS[act]
                if self.tile_free(a[0] + dx, a[1] + dy):
                    a[0] += dx
                    a[1] += dy
            elif act == "BOMB" and a[2]:
                self.bombs.append([a[0], a[1], BOMB_TIMER, name])
                a[2] = False
            # WAIT / None / invalid: nothing happens

        # update_explosions
        remaining = []
        for ex in self.explosions:
            ex[1] -= 1
            if ex[1] <= 0:
                ex[2] += 1
                if ex[2] == 1:
                    ex[1] = EXPLOSION_TIMER          # len(Explosion.ASSETS[1])
                    # The owner may have died before its own blast cleared, in
                    # which case it is not in this (self-centred) state at all.
                    if ex[3] in self.agents:
                        self.agents[ex[3]][2] = True
                else:
                    ex[2] = None
            if ex[2] is not None:
                remaining.append(ex)
        self.explosions = remaining

        # update_bombs
        still = []
        for b in self.bombs:
            if b[2] <= 0:
                coords = blast_coords(b[0], b[1], self.arena)
                for (x, y) in coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                self.explosions.append([coords, EXPLOSION_TIMER, 0, b[3]])
            else:
                b[2] -= 1
                still.append(b)
        self.bombs = still

        # evaluate_explosions
        for ex in self.explosions:
            if ex[2] != 0:
                continue
            for name, a in self.agents.items():
                if not a[3] and (a[0], a[1]) in ex[0]:
                    a[3] = True


def state_from_snapshot(snap) -> State:
    """Rebuild a `State` from one recorded pre-step snapshot."""
    arena = np.array(snap["arena"], dtype=np.int8)
    agents = {"ME": [int(snap["self"][0]), int(snap["self"][1]),
                     bool(snap["bombs_left"]), False]}
    for name, (x, y), left in snap["others"]:
        agents[name] = [int(x), int(y), bool(left), False]
    bombs = [[int(p[0]), int(p[1]), int(t), ("ME" if o == "benedict_task4" else o)]
             for p, t, o in snap["bombs"]]
    explosions = [[[(int(x), int(y)) for x, y in coords], int(t), int(stage),
                   ("ME" if o == "benedict_task4" else o)]
                  for coords, t, stage, o in snap["explosions"]]
    return State(arena, bombs, explosions, agents, step=snap["step"] - 1)


def order_from_snapshot(snap) -> list[str]:
    """Engine activation order for that step: the recorded permutation."""
    names = ["ME" if n == "benedict_task4" else n for n in snap["active_order"]]
    return [names[int(i)] for i in snap["perm"]]


def actions_from_snapshot(snap, my_action=None) -> dict:
    acts = {("ME" if n == "benedict_task4" else n): a
            for n, a in snap["actions"].items()}
    if my_action is not None:
        acts["ME"] = my_action
    return acts


# ---------------------------------------------------------------------------
# Frozen-opponent survivability
# ---------------------------------------------------------------------------
SAFE_HORIZON = BOMB_TIMER + EXPLOSION_TIMER + 2


def lethal_schedule(state: State) -> list[set]:
    """`out[t]` = tiles that kill an agent standing there at the END of step t.

    t = 0 is the step about to be executed. Only bombs and explosions that
    already exist are counted -- opponents are frozen and drop nothing more,
    which is what makes this a *lower* bound on how doomed the agent was.
    """
    horizon = SAFE_HORIZON + 2
    out = [set() for _ in range(horizon)]

    for ex in state.explosions:
        if ex[2] != 0:
            continue
        # A stage-0 explosion seen with timer T is dangerous for T-1 more step
        # ends: `update_explosions` decrements before `evaluate_explosions`, so
        # T=1 has already had its last kill. Same convention as the agent's own
        # `danger_map`, which reads `explosion_map = timer - 1`.
        for t in range(ex[1] - 1):
            out[t].update(ex[0])

    for b in state.bombs:
        coords = blast_coords(b[0], b[1], state.arena)
        # timer `t` at the start of this step -> detonates at the end of step t
        for t in range(b[2], b[2] + EXPLOSION_TIMER):
            if t < horizon:
                out[t].update(coords)
    return out


def survivable(state: State, name: str = "ME") -> bool:
    """Is there ANY move sequence keeping `name` alive until the board is clear?

    Space-time BFS over free tiles. Opponents and bombs are static obstacles;
    crates are never cleared (conservative: an escape route that only opens
    after a blast is not counted).
    """
    lethal = lethal_schedule(state)
    horizon = max(
        [b[2] + EXPLOSION_TIMER for b in state.bombs]
        + [ex[1] - 1 for ex in state.explosions if ex[2] == 0] + [0]
    )
    if horizon == 0:
        return True

    arena = state.arena
    blocked = {(b[0], b[1]) for b in state.bombs}
    blocked |= {(a[0], a[1]) for n, a in state.agents.items()
                if not a[3] and n != name}

    me = state.agents[name]
    frontier = {(me[0], me[1])}
    for t in range(horizon):
        nxt = set()
        for (x, y) in frontier:
            for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if (dx or dy) and (arena[nx, ny] != 0 or (nx, ny) in blocked):
                    continue
                if (nx, ny) in lethal[t]:
                    continue
                nxt.add((nx, ny))
        if not nxt:
            return False
        frontier = nxt
    return True

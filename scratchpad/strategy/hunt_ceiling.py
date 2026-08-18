#!/usr/bin/env python3
"""The ceiling of Benedict's hunt strategy, measured without training anything.

Question: "while destroying crates and collecting coins, work one's way toward
the nearest opponent and try to kill it" -- is that worth points? A feature can
only ever approximate the strategy, so measure the strategy *played perfectly*
first. If an oracle that always knows where a trap is cannot beat the shipped
table, no digit encoding it will.

The policy is the shipped Q-table's greedy action, overridden only when a
*self-survivable trap site* for a reachable opponent lies within `k` steps: then
walk the BFS first step toward that site, and BOMB on arrival. A trap site is a
free tile whose bomb would leave that opponent with no survivable escape,
evaluated with the same time-aware BFS `callbacks.escape_direction` runs on
ourselves. `k = -1` disables the override and must reproduce the shipped agent.

Nothing under `agent_code/` is written or read as a policy: the line-up uses the
provided `agent_code/user_agent/`, whose `act` returns `game_state['user_input']`,
and this script supplies that input. The shipped table is loaded read-only and
`callbacks.state_to_features` is called directly, so the feature map can never
drift from the agent's own.

The evaluation protocol mirrors `tools/evaluate.py` exactly -- per-round
`world.rng` reseed *and* the `np.random.seed` that reaches the provided
opponents -- so the CSV it writes is paired, arena for arena, with every other
rung-4 evaluation and can be fed straight to `tools/analyze.py --compare`.

    uv run python scratchpad/strategy/hunt_ceiling.py --k -1 --n-rounds 300
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("BM_QUIET_LOGS", "1")

import settings as s                                          # noqa: E402
from environment import BombeRLeWorld, WorldArgs              # noqa: E402
import agent_code.benedict_task4.callbacks as cb              # noqa: E402

DELTAS = cb.DELTAS
SAFE = cb.SAFE

FRAMEWORK_STATS = ["score", "coins", "kills", "suicides", "crates", "bombs",
                   "moves", "invalid", "steps", "time"]


# --------------------------------------------------------------------------
# Geometry shared with the diagnostic probes in this directory
# --------------------------------------------------------------------------
def dist_map(sx: int, sy: int, field: np.ndarray, blocked: set) -> np.ndarray:
    """BFS distance from (sx, sy) over free tiles; -1 where unreachable."""
    d = np.full(field.shape, -1, dtype=np.int16)
    d[sx, sy] = 0
    queue = [(sx, sy)]
    head = 0
    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if d[nx, ny] != -1 or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            d[nx, ny] = d[cx, cy] + 1
            queue.append((nx, ny))
    return d


def escapable(sx: int, sy: int, field: np.ndarray, danger: np.ndarray,
              blocked: set) -> bool:
    """Can an agent at (sx, sy) reach a tile no blast covers, in time?

    Same timing rule as `callbacks.escape_direction`: standing on a tile at the
    end of step `depth` is fatal if the blast reaches it at or before `depth`.
    """
    if danger[sx, sy] >= SAFE:
        return True
    queue = [((sx, sy), 0)]
    seen = {(sx, sy)}
    head = 0
    while head < len(queue):
        (cx, cy), depth = queue[head]
        head += 1
        if depth >= SAFE:
            continue
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= SAFE:
                return True
            seen.add((nx, ny))
            queue.append(((nx, ny), depth + 1))
    return False


def trap_sites(p: tuple[int, int], field: np.ndarray, danger: np.ndarray,
               bombs: set) -> list[tuple[int, int]]:
    """Free tiles whose bomb traps the agent at `p` and that we could survive.

    Only tiles within `BOMB_POWER` of `p` can reach it at all, and the blast
    must actually cover `p` -- a stone wall in between stops it.
    """
    out = []
    px, py = p
    for bx in range(max(1, px - s.BOMB_POWER), min(field.shape[0], px + s.BOMB_POWER + 1)):
        for by in range(max(1, py - s.BOMB_POWER), min(field.shape[1], py + s.BOMB_POWER + 1)):
            if abs(bx - px) + abs(by - py) > s.BOMB_POWER:
                continue
            if field[bx, by] != 0 or (bx, by) in bombs:
                continue
            blast = cb.blast_coords(bx, by, field)
            if p not in blast:
                continue
            hyp = danger.copy()
            for (cx, cy) in blast:
                if s.BOMB_TIMER < hyp[cx, cy]:
                    hyp[cx, cy] = s.BOMB_TIMER
            if escapable(px, py, field, hyp, bombs | {(bx, by)}):
                continue                      # they walk out: not a trap
            if not escapable(bx, by, field, hyp, bombs):
                continue                      # we could not survive setting it
            out.append((bx, by))
    return out


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------
class HuntCeiling:
    """Shipped greedy policy plus an oracle hunt override within `k` steps."""

    def __init__(self, q: np.ndarray, k: int):
        self.q = q
        self.k = k
        # Byte-for-byte the tie-break in `callbacks.act`, including its seed, so
        # the k = -1 arm is the shipped agent and not merely something like it.
        self.policy_rng = np.random.default_rng(cb.POLICY_SEED)
        self.hunt_steps = 0
        self.hunt_bombs = 0
        self.steps = 0

    def greedy(self, game_state: dict) -> str:
        row = self.q[cb.state_to_features(game_state)]
        best = np.flatnonzero(row >= row.max() - cb.TIE_TOL)
        return cb.ACTIONS[int(best[0] if best.size == 1
                              else self.policy_rng.choice(best))]

    def act(self, game_state: dict) -> str:
        self.steps += 1
        if self.k < 0:
            return self.greedy(game_state)

        x, y = game_state["self"][3]
        field = game_state["field"]
        danger = cb.danger_map(game_state)

        # Never hunt while standing in a blast: the escape is worth more than any
        # kill, and overriding here would measure our escape logic, not the hunt.
        if danger[x, y] < SAFE or not game_state["self"][2]:
            return self.greedy(game_state)

        others = [o[3] for o in game_state["others"]]
        if not others:
            return self.greedy(game_state)

        bombs = {p for p, _ in game_state["bombs"]}
        # Other agents block movement (environment.py:121-126) but are not walls
        # for the search that finds *them*, so they are excluded from `blocked`
        # here and handled by the step-legality check below.
        dm = dist_map(x, y, field, bombs)

        # Cheap gate: no site within k can exist if the opponent itself is
        # further than k + BOMB_POWER away.
        reach = [(int(dm[p]), p) for p in others
                 if dm[p] >= 0 and dm[p] <= self.k + s.BOMB_POWER]
        if not reach:
            return self.greedy(game_state)

        best = None
        for _, p in sorted(reach):
            for site in trap_sites(p, field, danger, bombs):
                d = int(dm[site])
                if d < 0 or d > self.k:
                    continue
                if best is None or d < best[0]:
                    best = (d, site)
            if best is not None and best[0] == 0:
                break
        if best is None:
            return self.greedy(game_state)

        d, site = best
        if d == 0:
            self.hunt_bombs += 1
            return "BOMB"

        # Walk the first step of a shortest path to the site. Distances are taken
        # *from the site* so the descent is unambiguous; `dm` only ranked the
        # candidates. The step must be legal and not lethal -- an oracle that
        # walks into a blast would measure our recklessness, not the strategy.
        ds = dist_map(site[0], site[1], field, bombs)
        occupied = bombs | {o[3] for o in game_state["others"]}
        for action_idx, (dx, dy) in enumerate(DELTAS):
            nx, ny = x + dx, y + dy
            if field[nx, ny] != 0 or (nx, ny) in occupied:
                continue
            if danger[nx, ny] <= 0:
                continue
            if ds[nx, ny] == d - 1:
                self.hunt_steps += 1
                return cb.ACTIONS[action_idx]
        return self.greedy(game_state)


# --------------------------------------------------------------------------
# Evaluation loop -- mirrors tools/evaluate.py
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, required=True,
                    help="max walk to a trap site; -1 disables the override")
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=990731)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", default="results/eval/task4_tournament")
    a = ap.parse_args()

    label = a.label or f"benedict_huntceil_k{a.k}__task4_rb_{a.seed}"
    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    policy = HuntCeiling(q, a.k)

    log_dir = REPO / "logs" / "hunt_ceiling"
    log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=str(log_dir), save_stats=False,
        match_name=label, seed=a.seed, silence_errors=False, scenario="classic",
    )
    line_up = ["user_agent"] + ["rule_based_agent"] * 3
    world = BombeRLeWorld(args, [(n, False) for n in line_up])
    world.user_input = None

    records: list[dict] = []
    started = time.time()
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)          # reaches the provided opponents
        world.new_round()

        while world.running:
            me = [ag for ag in world.active_agents if ag.code_name == "user_agent"]
            action = "WAIT"
            if me:
                gs = world.get_state_for_agent(me[0])
                if gs is not None:
                    action = policy.act(gs)
            world.do_step(action)

        round_records = []
        for slot, agent in enumerate(world.agents):
            rec = {"round": r, "seed": a.seed + r, "slot": slot,
                   "agent": agent.name, "code": agent.code_name,
                   "survived": int(not agent.dead), "round_steps": world.step}
            for key in FRAMEWORK_STATS:
                rec[key] = agent.statistics.get(key, 0)
            died = int(agent.dead)
            rec["died"] = died
            rec["killed_by_opponent"] = max(0, died - rec["suicides"])
            round_records.append(rec)
        best = max(x["score"] for x in round_records)
        for rec in round_records:
            rec["rank"] = 1 + sum(1 for o in round_records if o["score"] > rec["score"])
            rec["won"] = int(rec["score"] == best)
        records.extend(round_records)
        if (r + 1) % 50 == 0:
            print(f"  {r+1}/{a.n_rounds}  {time.time()-started:.0f}s", flush=True)
    world.end()

    out = REPO / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{label}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    meta = {"label": label, "k": a.k, "seed": a.seed, "n_rounds": a.n_rounds,
            "line_up": line_up, "scenario": "classic",
            "table": "agent_code/benedict_task4/q_table.npy",
            "hunt_steps": policy.hunt_steps, "hunt_bombs": policy.hunt_bombs,
            "policy_steps": policy.steps, "wall_clock_s": round(time.time() - started, 1),
            "note": "ceiling probe, not a trained agent; user_agent driven by "
                    "scratchpad/strategy/hunt_ceiling.py"}
    with open(out / f"{label}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    us = [x for x in records if x["code"] == "user_agent"]
    m = lambda key: float(np.mean([x[key] for x in us]))
    print(f"\nk={a.k}  n={len(us)}  score {m('score'):.3f}  coins {m('coins'):.3f}  "
          f"kills {m('kills'):.3f}  won {m('won'):.3f}  suicides {m('suicides'):.3f}  "
          f"crates {m('crates'):.2f}  bombs {m('bombs'):.2f}  survived {m('survived'):.3f}")
    print(f"override fired on {policy.hunt_steps} moves + {policy.hunt_bombs} bombs "
          f"of {policy.steps} steps "
          f"({(policy.hunt_steps+policy.hunt_bombs)/max(policy.steps,1):.2%})")
    print(f"wrote {out / (label + '.csv')}")


if __name__ == "__main__":
    main()

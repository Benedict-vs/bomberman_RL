#!/usr/bin/env python3
"""Record every step leading up to a death of `benedict_task4` in a rung-4 field.

Drives `BombeRLeWorld` exactly the way `tools/evaluate.py` does (same per-round
reseeding of `world.rng` *and* the legacy global RNG) so the arenas match every
other measurement at the same base seed.  Nothing in the framework is edited;
the only extension is a subclass that notes which explosion covered our agent's
tile at the moment it was removed, because `evaluate_explosions` throws that
information away.

Output: one pickle with, per death, the last `WINDOW` steps as our agent saw
them, plus the recorded actions of every agent (needed for the frozen-opponent
counterfactual in `analyse.py`).
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import settings as s  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import candidates  # noqa: E402

AGENT = "benedict_task4"
WINDOW = 10          # snapshots kept before the death step


def load_callbacks(agent: str):
    """Import the agent's own feature code, so digits are computed by it, not by us."""
    path = REPO_ROOT / "agent_code" / agent / "callbacks.py"
    spec = importlib.util.spec_from_file_location(f"{agent}_callbacks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ForensicWorld(BombeRLeWorld):
    """`BombeRLeWorld` that remembers which explosion killed our agent."""

    def __init__(self, args, agents):
        super().__init__(args, agents)
        self.me = next(a for a in self.agents if a.code_name == AGENT)
        self.killers: list[tuple] = []

    def evaluate_explosions(self):
        me = self.me
        if not me.dead and me in self.active_agents:
            for ex in self.explosions:
                if ex.is_dangerous() and (me.x, me.y) in ex.blast_coords:
                    # blast_coords[0] is the bomb tile (items.py:Bomb.get_blast_coords)
                    self.killers.append((ex.owner.name, ex.blast_coords[0],
                                         tuple(ex.blast_coords), self.step))
        super().evaluate_explosions()


def world_args(seed: int, log_dir: Path) -> WorldArgs:
    return WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=str(log_dir), save_stats=False,
        match_name="deaths", seed=seed, silence_errors=False, scenario="classic",
    )


def snapshot(world, cb) -> dict:
    """Everything about a single step, from our agent's point of view."""
    me = world.me
    gs = world.get_state_for_agent(me)
    digits = feature_digits(gs, cb)
    cand = candidates.compute(gs)
    return {
        "cand": cand,
        "step": world.step + 1,          # the step about to be executed
        "arena": np.array(world.arena, dtype=np.int8),
        "self": (me.x, me.y),
        "bombs_left": me.bombs_left,
        "score": me.score,
        "others": [(o.name, (o.x, o.y), o.bombs_left)
                   for o in world.active_agents if o is not me],
        "bombs": [((b.x, b.y), b.timer, b.owner.name) for b in world.bombs],
        "explosions": [(tuple(ex.blast_coords), ex.timer, ex.stage, ex.owner.name)
                       for ex in world.explosions],
        "coins": [(c.x, c.y) for c in world.coins if c.collectable],
        "digits": digits,
        "row": cb.encode(digits),
        "active_order": [a.name for a in world.active_agents],
        "crates_left": int((world.arena == 1).sum()),
    }


def feature_digits(gs, cb) -> tuple:
    """The eight digits, recomputed by the agent's own code from its own state."""
    field = gs["field"]
    x, y = gs["self"][3]
    have_bomb = gs["self"][2]
    danger = cb.danger_map(gs)
    occupied = {pos for pos, _ in gs["bombs"]}
    occupied.update(o[3] for o in gs["others"])
    own_danger = 0 if danger[x, y] >= cb.SAFE else int(danger[x, y]) + 1
    others = [o[3] for o in gs["others"]]
    bomb_useful = int(have_bomb and cb.bomb_hits_crate(x, y, field, others))
    if own_danger:
        target = cb.escape_direction(x, y, field, danger, occupied)
        target_dist = cb.DIST_NONE
    else:
        target, distance = cb.target_direction(x, y, field, gs["coins"], others)
        target_dist = cb.distance_bucket(distance)
    return cb.neighbour_status(x, y, field, danger, occupied) + (
        own_danger, target, bomb_useful, target_dist)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "deaths.pkl")
    args = ap.parse_args()

    cb = load_callbacks(AGENT)
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    line_up = [(AGENT, False)] + [("rule_based_agent", False)] * 3
    world = ForensicWorld(world_args(args.seed, log_dir), line_up)
    me = world.me

    deaths, rounds, step_log = [], [], []
    digit_counts = {}          # base rates over every step our agent was alive
    started = time.time()

    for r in range(args.n_rounds):
        world.rng = np.random.default_rng(args.seed + r)
        np.random.seed(args.seed + r)
        world.new_round()
        world.killers = []
        world.user_input = None      # do_step sets it; we peek at the state first

        window: deque = deque(maxlen=WINDOW)
        n_alive_steps = 0

        while world.running:
            snap = None
            if not me.dead and me in world.active_agents:
                snap = snapshot(world, cb)
            world.do_step()
            if snap is not None:
                # `actions` desyncs the moment an agent dies (its list stops
                # growing), so only names still in active_order at snapshot time
                # are read. `rule_based_agent.act` genuinely returns None when it
                # has no valid action -- the framework then scores it INVALID.
                acts = {}
                for name in snap["active_order"]:
                    lst = world.replay["actions"][name]
                    acts[name] = lst[-1] if lst else None
                snap["actions"] = acts
                snap["my_action"] = world.replay["actions"][me.name][-1]
                snap["perm"] = list(world.replay["permutations"][-1])
                window.append(snap)
                n_alive_steps += 1
                digit_counts[snap["digits"]] = digit_counts.get(snap["digits"], 0) + 1
                step_log.append({
                    "round": r, "step": snap["step"], "digits": snap["digits"],
                    "row": snap["row"], "cand": snap["cand"],
                    "action": snap["my_action"],
                    "n_bombs": len(snap["bombs"]),
                    "crates_left": snap["crates_left"],
                })
            if me.dead and not any(d["round"] == r for d in deaths):
                deaths.append({
                    "round": r,
                    "seed": args.seed + r,
                    "death_step": world.step,
                    "killers": list(world.killers),
                    "window": list(window),
                    "score": me.score,
                    "suicide": me.statistics.get("suicides", 0) > 0,
                })

        rounds.append({
            "round": r, "seed": args.seed + r, "steps": world.step,
            "died": int(me.dead), "suicides": me.statistics.get("suicides", 0),
            "score": me.score, "kills": me.statistics.get("kills", 0),
            "coins": me.statistics.get("coins", 0),
            "crates": me.statistics.get("crates", 0),
            "alive_steps": n_alive_steps,
            "death_step": next((d["death_step"] for d in deaths if d["round"] == r), None),
        })

        if (r + 1) % 25 == 0:
            el = time.time() - started
            print(f"  round {r+1}/{args.n_rounds}  deaths={len(deaths)}  "
                  f"{el:.0f}s elapsed", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump({"deaths": deaths, "rounds": rounds,
                     "digit_counts": digit_counts, "step_log": step_log,
                     "seed": args.seed, "n_rounds": args.n_rounds}, f)
    print(f"wrote {args.out}: {len(deaths)} deaths in {args.n_rounds} rounds")


if __name__ == "__main__":
    main()

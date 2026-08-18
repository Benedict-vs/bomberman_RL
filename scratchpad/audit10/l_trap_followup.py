#!/usr/bin/env python3
"""Audit 10 -- does the oracle's "inescapable trap" actually kill?

hunt_ceiling.py's whole claim to being an *upper bound* rests on `trap_sites` identifying
bomb placements the target cannot survive. The CSVs only show the net score effect, so this
copy of the harness records, for every oracle BOMB, whether the targeted opponent is dead
within EXPLOSION_TIMER + BOMB_TIMER steps and whether WE got the kill credit.

Nothing outside scratchpad/audit10/ is written. Policy code is imported from the original
probe so the trap definition cannot drift.
"""
from __future__ import annotations
import os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad/strategy"))

import settings as s                                        # noqa: E402
from environment import BombeRLeWorld, WorldArgs            # noqa: E402
import agent_code.benedict_task4.callbacks as cb            # noqa: E402
from hunt_ceiling import HuntCeiling, trap_sites, dist_map  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = 990731


class Traced(HuntCeiling):
    """Same policy; records the target whenever the override drops a bomb."""

    def __init__(self, q, k):
        super().__init__(q, k)
        self.last_bomb_target: tuple[int, int] | None = None

    def act(self, game_state):
        before = self.hunt_bombs
        self.last_bomb_target = None
        # recompute the target the same way the parent does, but only when it bombs
        action = super().act(game_state)
        if self.hunt_bombs > before:
            x, y = game_state["self"][3]
            field = game_state["field"]
            danger = cb.danger_map(game_state)
            bombs = {p for p, _ in game_state["bombs"]}
            dm = dist_map(x, y, field, bombs)
            for o in game_state["others"]:
                p = o[3]
                if dm[p] < 0 or dm[p] > self.k + s.BOMB_POWER:
                    continue
                if (x, y) in trap_sites(p, field, danger, bombs):
                    self.last_bomb_target = o[0]
                    break
        return action


q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = Traced(q, K)
log_dir = REPO / "logs" / "audit10"
log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                 match_name="audit10_trace", seed=SEED, silence_errors=False,
                 scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None

FUSE = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1
c = Counter()
for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r)
    np.random.seed(SEED + r)
    world.new_round()
    pending: list[tuple[str, int, float]] = []   # (target, deadline, our kills at drop)
    me = [a for a in world.agents if a.code_name == "user_agent"][0]
    while world.running:
        alive = [a for a in world.active_agents if a.code_name == "user_agent"]
        action = "WAIT"
        if alive:
            gs = world.get_state_for_agent(alive[0])
            if gs is not None:
                action = policy.act(gs)
        if policy.last_bomb_target is not None:
            c["oracle_bombs"] += 1
            pending.append((policy.last_bomb_target, world.step + FUSE,
                            me.statistics.get("kills", 0)))
        world.do_step(action)
        still = [(t, dl, k0) for (t, dl, k0) in pending if world.step < dl]
        for (t, dl, k0) in pending:
            if (t, dl, k0) in still:
                continue
            dead = any(a.name == t and a.dead for a in world.agents)
            c["target_dead" if dead else "target_alive"] += 1
            if dead and me.statistics.get("kills", 0) > k0:
                c["we_got_credit"] += 1
        pending = still
    for (t, dl, k0) in pending:                       # round ended before the fuse ran out
        dead = any(a.name == t and a.dead for a in world.agents)
        c["target_dead" if dead else "target_alive"] += 1
        if dead and me.statistics.get("kills", 0) > k0:
            c["we_got_credit"] += 1
    if (r + 1) % 50 == 0:
        print(f"  {r+1}/{ROUNDS}", flush=True)
world.end()

n = c["oracle_bombs"]
print(f"\nk={K}  rounds={ROUNDS}")
print(f"oracle 'inescapable trap' bombs placed : {n}  ({n/ROUNDS:.3f} per round)")
print(f"  target dead within the fuse          : {c['target_dead']} "
      f"({c['target_dead']/max(n,1):.1%})")
print(f"  ... and WE were credited with a kill : {c['we_got_credit']} "
      f"({c['we_got_credit']/max(n,1):.1%})")
print(f"  target still alive                   : {c['target_alive']} "
      f"({c['target_alive']/max(n,1):.1%})")

#!/usr/bin/env python3
"""E40 P3 -- do the corrected oracle's bombs actually kill what they aim at?

P3 pre-registers that the simultaneous-move trap test produces *better* bombs:
a credited-kill rate materially above the 8.1 % audit 10 measured for the stale
test, with >= 20 % registered as the bar. This follows every override bomb to
its fuse and asks whether the targeted opponent died and who got the credit.

Method follows scratchpad/audit10/l_trap_followup.py, re-pointed at
hunt_ceiling_v2 so both trap models can be traced from one binary. Policy code
is imported, never copied, so the trap definition cannot drift from the sweep's.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s                                            # noqa: E402
from environment import BombeRLeWorld, WorldArgs                # noqa: E402
import agent_code.benedict_task4.callbacks as cb                # noqa: E402
from hunt_ceiling_v2 import HuntCeiling, trap_sites, dist_map   # noqa: E402

TRAP_MODEL = sys.argv[1] if len(sys.argv) > 1 else "sim"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEED = 990731
SIM = TRAP_MODEL == "sim"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- honest at the small n these traces produce."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class Traced(HuntCeiling):
    """Same policy; records which opponent the override was aiming at."""

    def __init__(self, q, k, simultaneous):
        super().__init__(q, k, simultaneous)
        self.last_bomb_target: str | None = None

    def act(self, game_state):
        before = self.hunt_bombs
        self.last_bomb_target = None
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
                if (x, y) in trap_sites(p, field, danger, bombs, self.simultaneous):
                    self.last_bomb_target = o[0]
                    break
        return action


q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = Traced(q, K, SIM)
log_dir = REPO / "logs" / "e40_conversion"
log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir),
                 save_stats=False, match_name=f"e40_conv_{TRAP_MODEL}_k{K}",
                 seed=SEED, silence_errors=False, scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None

FUSE = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1
c: Counter = Counter()
for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r)
    np.random.seed(SEED + r)
    world.new_round()
    pending: list[tuple[str, int, float]] = []      # (target, deadline, our kills at drop)
    me = [a for a in world.agents if a.code_name == "user_agent"][0]

    def settle(item) -> None:
        target, _, kills_at_drop = item
        dead = any(a.name == target and a.dead for a in world.agents)
        c["target_dead" if dead else "target_alive"] += 1
        if dead and me.statistics.get("kills", 0) > kills_at_drop:
            c["we_got_credit"] += 1

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
        still = [it for it in pending if world.step < it[1]]
        for it in pending:
            if it not in still:
                settle(it)
        pending = still
    for it in pending:                              # round ended inside the fuse window
        settle(it)
    if (r + 1) % 200 == 0:
        print(f"  {r+1}/{ROUNDS}", flush=True)
world.end()

n = c["oracle_bombs"]
dead, credit = c["target_dead"], c["we_got_credit"]
print(f"\ntrap_model={TRAP_MODEL}  k={K}  rounds={ROUNDS}")
print(f"override bombs placed          : {n}  ({n / ROUNDS:.3f} per round)")
for name, hits in (("target dead within fuse", dead), ("WE credited with the kill", credit)):
    lo, hi = wilson(hits, n)
    print(f"  {name:<28}: {hits:>5} ({hits / max(n, 1):>6.1%})  95% Wilson [{lo:.1%}, {hi:.1%}]")
print(f"  credited kills per round     : {credit / ROUNDS:.4f}")

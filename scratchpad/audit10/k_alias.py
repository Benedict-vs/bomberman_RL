#!/usr/bin/env python3
"""Audit 10 -- is "bombing here is certain death" a CONSTANT, or is it ALIASED?

trap_opportunity.py reports an escape exists on 99.0 % of armed steps and NEXT_STEPS §4
converts that into "the bit is a constant, don't spend a digit on it". Marginal rarity is
not the test. The test is whether the shipped feature row already determines the bit: if
every row that ever carries "bombing here kills me" carries *only* that value, the digit is
redundant; if the same row carries both values, the digit is new information in exactly the
states where the agent's dominant failure mode lives.
"""
from __future__ import annotations
import os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO))
from environment import BombeRLeWorld          # noqa: E402
from fallbacks import pygame                   # noqa: E402,F401
import agent_code.benedict_task4.callbacks as cb   # noqa: E402
import settings as s                           # noqa: E402

US = "benedict_task4"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED = 550731


def escapable(sx, sy, field, danger, blocked, horizon):
    if danger[sx, sy] >= horizon:
        return True
    queue = [((sx, sy), 0)]; seen = {(sx, sy)}; head = 0
    while head < len(queue):
        (cx, cy), depth = queue[head]; head += 1
        if depth >= horizon:
            continue
        for dx, dy in cb.DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= horizon:
                return True
            seen.add((nx, ny)); queue.append(((nx, ny), depth + 1))
    return False


q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
args = type("A", (), {
    "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
    "save_replay": False, "replay": None, "make_video": False,
    "continue_without_training": True, "log_dir": str(REPO / "logs"),
    "save_stats": False, "match_name": "audit10_alias", "seed": SEED,
    "silence_errors": False, "scenario": "classic"})()
world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
world.user_input = None

row_safe: dict[int, Counter] = defaultdict(Counter)   # row -> {True: n_safe, False: n_deadly}
greedy_of_row: dict[int, str] = {}
n_armed = 0
for _ in range(ROUNDS):
    world.new_round()
    while world.running and world.step < 400:
        for ag in world.active_agents:
            if ag.code_name != US:
                continue
            gs = world.get_state_for_agent(ag)
            if gs is None or not gs["self"][2]:
                continue
            n_armed += 1
            x, y = gs["self"][3]
            field = gs["field"]
            hyp = cb.danger_map(gs)
            for (bx, by) in cb.blast_coords(x, y, field):
                hyp[bx, by] = min(hyp[bx, by], s.BOMB_TIMER)
            ok = escapable(x, y, field, hyp, {p for p, _ in gs["bombs"]}, cb.SAFE)
            idx = int(cb.state_to_features(gs))
            row_safe[idx][bool(ok)] += 1
            row = q[idx]
            best = np.flatnonzero(row >= row.max() - cb.TIE_TOL)
            greedy_of_row[idx] = cb.ACTIONS[int(best[0])] if best.size == 1 else "TIE"
        world.do_step("WAIT")
world.end()

deadly_steps = sum(c[False] for c in row_safe.values())
print(f"rounds {ROUNDS}   armed steps {n_armed}")
print(f"'bombing here is certain death' on {deadly_steps} armed steps "
      f"({deadly_steps/n_armed:.4%}) = {deadly_steps/ROUNDS:.2f} per round")
print(f"distinct feature rows seen while armed: {len(row_safe)}")

pure_deadly = [r for r, c in row_safe.items() if c[False] and not c[True]]
mixed = [r for r, c in row_safe.items() if c[False] and c[True]]
print(f"rows that are ALWAYS deadly-to-bomb: {len(pure_deadly)} "
      f"({sum(row_safe[r][False] for r in pure_deadly)} steps)")
print(f"rows that carry BOTH values (aliased): {len(mixed)} "
      f"({sum(row_safe[r][False] for r in mixed)} deadly steps, "
      f"{sum(row_safe[r][True] for r in mixed)} safe steps)")
al = sum(row_safe[r][False] for r in mixed)
print(f"-> {al/max(deadly_steps,1):.1%} of the deadly-to-bomb steps sit in a row that "
      f"ALSO contains safe-to-bomb steps; the shipped digits cannot tell them apart.")

print("\nwhat the shipped table says in the aliased rows, weighted by deadly steps:")
acts = Counter()
for r in mixed:
    acts[greedy_of_row[r]] += row_safe[r][False]
tot = sum(acts.values()) or 1
print("  " + "  ".join(f"{a}:{n/tot:.3f}" for a, n in acts.most_common()))
print("\nsame for the always-deadly rows:")
acts = Counter()
for r in pure_deadly:
    acts[greedy_of_row[r]] += row_safe[r][False]
tot = sum(acts.values()) or 1
print("  " + "  ".join(f"{a}:{n/tot:.3f}" for a, n in acts.most_common()))

# conditional risk: among deadly-to-bomb states, how often would the agent bomb?
bomb_deadly = sum(row_safe[r][False] for r in row_safe if greedy_of_row[r] in ("BOMB", "TIE"))
print(f"\ndeadly-to-bomb steps where the greedy action is BOMB (or an untrained tie): "
      f"{bomb_deadly} = {bomb_deadly/ROUNDS:.3f} per round, against a measured "
      f"suicide rate of 0.488 per round")

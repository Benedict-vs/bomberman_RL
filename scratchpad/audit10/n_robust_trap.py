#!/usr/bin/env python3
"""Audit 10 -- is the low conversion the STRATEGY's fault or the TRAP MODEL's?

environment.py:420-427 hands every agent the same pre-action snapshot and only then executes
the actions in a random permutation. So at the instant our BOMB lands, the target has already
taken one move that hunt_ceiling.trap_sites never modelled: it evaluates the trap against the
tile the opponent is *leaving*.

This arm keeps hunt_ceiling's policy skeleton and replaces only `trap_sites` with a
simultaneous-move version: a site counts only if the target is caught and escape-less from
its current tile AND from every tile it could legally step to this turn. Then measure the same
conversion rate as l_trap_followup.py. If the conversion jumps, the ceiling measured the model,
not the strategy.
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

import settings as s                                     # noqa: E402
from environment import BombeRLeWorld, WorldArgs         # noqa: E402
import agent_code.benedict_task4.callbacks as cb         # noqa: E402
import hunt_ceiling as hc                                # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = 990731


def robust_trap_sites(p, field, danger, bombs):
    """hunt_ceiling.trap_sites, but the target is allowed its simultaneous move."""
    px, py = p
    spots = [p] + [(px + dx, py + dy) for dx, dy in cb.DELTAS
                   if field[px + dx, py + dy] == 0 and (px + dx, py + dy) not in bombs]
    out = []
    for bx in range(max(1, px - s.BOMB_POWER - 1), min(field.shape[0], px + s.BOMB_POWER + 2)):
        for by in range(max(1, py - s.BOMB_POWER - 1), min(field.shape[1], py + s.BOMB_POWER + 2)):
            if field[bx, by] != 0 or (bx, by) in bombs:
                continue
            blast = set(cb.blast_coords(bx, by, field))
            if not all(q in blast for q in spots):
                continue
            hyp = danger.copy()
            for (cx, cy) in blast:
                if s.BOMB_TIMER < hyp[cx, cy]:
                    hyp[cx, cy] = s.BOMB_TIMER
            if any(hc.escapable(qx, qy, field, hyp, bombs | {(bx, by)}) for qx, qy in spots):
                continue
            if not hc.escapable(bx, by, field, hyp, bombs):
                continue
            out.append((bx, by))
    return out


class Robust(hc.HuntCeiling):
    def __init__(self, q, k):
        super().__init__(q, k)
        self.last_bomb_target = None

    def act(self, game_state):
        self.last_bomb_target = None
        self.steps += 1
        if self.k < 0:
            return self.greedy(game_state)
        x, y = game_state["self"][3]
        field = game_state["field"]
        danger = cb.danger_map(game_state)
        if danger[x, y] < cb.SAFE or not game_state["self"][2]:
            return self.greedy(game_state)
        others = game_state["others"]
        if not others:
            return self.greedy(game_state)
        bombs = {p for p, _ in game_state["bombs"]}
        dm = hc.dist_map(x, y, field, bombs)
        reach = [(int(dm[o[3]]), o) for o in others
                 if dm[o[3]] >= 0 and dm[o[3]] <= self.k + s.BOMB_POWER]
        if not reach:
            return self.greedy(game_state)
        best = None
        for _, o in sorted(reach, key=lambda t: t[0]):
            for site in robust_trap_sites(o[3], field, danger, bombs):
                d = int(dm[site])
                if d < 0 or d > self.k:
                    continue
                if best is None or d < best[0]:
                    best = (d, site, o[0])
            if best is not None and best[0] == 0:
                break
        if best is None:
            return self.greedy(game_state)
        d, site, target = best
        if d == 0:
            self.hunt_bombs += 1
            self.last_bomb_target = target
            return "BOMB"
        ds = hc.dist_map(site[0], site[1], field, bombs)
        occupied = bombs | {o[3] for o in others}
        for ai, (dx, dy) in enumerate(cb.DELTAS):
            nx, ny = x + dx, y + dy
            if field[nx, ny] != 0 or (nx, ny) in occupied:
                continue
            if danger[nx, ny] <= 0:
                continue
            if ds[nx, ny] == d - 1:
                self.hunt_steps += 1
                return cb.ACTIONS[ai]
        return self.greedy(game_state)


q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = Robust(q, K)
log_dir = REPO / "logs" / "audit10"
log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                 match_name="audit10_robust", seed=SEED, silence_errors=False,
                 scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None

FUSE = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1
c = Counter()
scores = []
for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r)
    np.random.seed(SEED + r)
    world.new_round()
    me = [a for a in world.agents if a.code_name == "user_agent"][0]
    pending = []
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
        still = [t for t in pending if world.step < t[1]]
        for t in pending:
            if t in still:
                continue
            dead = any(a.name == t[0] and a.dead for a in world.agents)
            c["target_dead" if dead else "target_alive"] += 1
            if dead and me.statistics.get("kills", 0) > t[2]:
                c["we_got_credit"] += 1
        pending = still
    for t in pending:
        dead = any(a.name == t[0] and a.dead for a in world.agents)
        c["target_dead" if dead else "target_alive"] += 1
        if dead and me.statistics.get("kills", 0) > t[2]:
            c["we_got_credit"] += 1
    scores.append((me.statistics.get("score", 0), me.statistics.get("kills", 0)))
    if (r + 1) % 50 == 0:
        print(f"  {r+1}/{ROUNDS}", flush=True)
world.end()

n = max(c["oracle_bombs"], 1)
sc = np.array(scores, float)
print(f"\nROBUST (simultaneous-move) trap model, k={K}, rounds={ROUNDS}")
print(f"  override bombs                 : {c['oracle_bombs']} ({c['oracle_bombs']/ROUNDS:.3f}/round)"
      f"   walk-steps {policy.hunt_steps}")
print(f"  target dead within the fuse    : {c['target_dead']} ({c['target_dead']/n:.1%})")
print(f"  we were credited with the kill : {c['we_got_credit']} ({c['we_got_credit']/n:.1%})")
print(f"  score {sc[:,0].mean():.3f}   kills {sc[:,1].mean():.3f}")

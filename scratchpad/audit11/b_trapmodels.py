"""Audit 11 -- how many trap sites do the competing trap models find on the SAME states?

Runs the control policy (k = -1, i.e. the shipped table, no override) so every
model is scored on one common state distribution, and at each step evaluates:

  stale     e40 `--trap-model stale`   (E39 behaviour)
  sim       e40 `--trap-model sim`     (E40's "corrected" model, as shipped)
  sim_clk   sim with the fuse clock fixed: the target is evaluated from cells it
            occupies at the END of the step, i.e. at the snapshot one step later,
            where our bomb's timer is BOMB_TIMER-1 = 3, not BOMB_TIMER = 4.
            danger_map's own docstring: "a bomb the agent sees at timer t kills
            at the end of step now + t".
  sim_clk2  sim_clk plus the same -1 applied to every pre-existing bomb in the
            copied danger map (they are also one step stale for a post-move cell)
  a10       audit 10's n_robust_trap.robust_trap_sites: additionally requires the
            target's current tile AND every step-to tile to lie inside OUR blast

Reports, per step where the cheap gate passes, whether each model would fire
(site at distance 0 -> BOMB) or walk (site within k).
"""
from __future__ import annotations
import os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s
from environment import BombeRLeWorld, WorldArgs
import agent_code.benedict_task4.callbacks as cb
from hunt_ceiling_v2 import HuntCeiling, dist_map, escapable, target_cells

K = 4
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 990731


def sites(p, field, danger, bombs, mode):
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
            new_val = s.BOMB_TIMER
            if mode in ("sim_clk", "sim_clk2"):
                new_val = s.BOMB_TIMER - 1
            if mode == "sim_clk2":
                hyp = np.where(hyp < cb.SAFE, np.maximum(hyp - 1, 0), hyp).astype(np.int8)
            for (cx, cy) in blast:
                if new_val < hyp[cx, cy]:
                    hyp[cx, cy] = new_val
            cells = [p] if mode == "stale" else target_cells(p, field, bombs | {(bx, by)})
            if mode == "a10":
                bs = set(blast)
                if not all(q in bs for q in cells):
                    continue
            if any(escapable(cx, cy, field, hyp, bombs | {(bx, by)}) for cx, cy in cells):
                continue
            # our own survivability -- unchanged from e40 (correct: we evaluate
            # from our own pre-move tile, where BOMB_TIMER is the right clock)
            own = danger.copy()
            for (cx, cy) in blast:
                if s.BOMB_TIMER < own[cx, cy]:
                    own[cx, cy] = s.BOMB_TIMER
            if not escapable(bx, by, field, own, bombs):
                continue
            out.append((bx, by))
    return out


MODES = ["stale", "sim", "sim_clk", "sim_clk2", "a10"]

q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = HuntCeiling(q, -1)
log_dir = REPO / "logs" / "audit11"; log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                 match_name="audit11_trapmodels", seed=SEED, silence_errors=False,
                 scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None

c = Counter()
for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r)
    np.random.seed(SEED + r)
    world.new_round()
    while world.running:
        alive = [a for a in world.active_agents if a.code_name == "user_agent"]
        action = "WAIT"
        if alive:
            gs = world.get_state_for_agent(alive[0])
            if gs is not None:
                action = policy.act(gs)
                c["steps"] += 1
                x, y = gs["self"][3]
                field = gs["field"]
                danger = cb.danger_map(gs)
                if danger[x, y] >= cb.SAFE and gs["self"][2] and gs["others"]:
                    bombs = {p for p, _ in gs["bombs"]}
                    dm = dist_map(x, y, field, bombs)
                    reach = [p for p in (o[3] for o in gs["others"])
                             if dm[p] >= 0 and dm[p] <= K + s.BOMB_POWER]
                    if reach:
                        c["gate"] += 1
                        for mode in MODES:
                            best = None
                            for p in reach:
                                for st in sites(p, field, danger, bombs, mode):
                                    d = int(dm[st])
                                    if d < 0 or d > K:
                                        continue
                                    if best is None or d < best:
                                        best = d
                            if best is not None:
                                c[f"{mode}_any"] += 1
                                if best == 0:
                                    c[f"{mode}_bomb"] += 1
        world.do_step(action)
    if (r + 1) % 50 == 0:
        print(f"  {r+1}/{ROUNDS}", flush=True)
world.end()

print(f"\ncontrol rollout, {ROUNDS} rounds, {c['steps']} of our steps, gate passed {c['gate']}")
print(f"{'model':<10}{'would-walk-or-bomb':>20}{'would-BOMB':>12}{'BOMB/round':>12}{'vs sim':>10}")
base = c["sim_bomb"]
for m in MODES:
    print(f"{m:<10}{c[m+'_any']:>20}{c[m+'_bomb']:>12}{c[m+'_bomb']/ROUNDS:>12.3f}"
          f"{(c[m+'_bomb']/max(base,1)):>10.2f}x")

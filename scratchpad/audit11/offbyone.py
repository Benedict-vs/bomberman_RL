"""Audit 11 F3: is hunt_ceiling_v2's "simultaneous move" trap test correctly timed?

Timing, from environment.py:
  step T: self.step+=1 -> poll_and_run_agents (snapshot, then execute in a random
  permutation) -> collect_coins -> update_explosions -> update_bombs -> evaluate_explosions.
  A bomb placed in step T has timer BOMB_TIMER=4; update_bombs decrements it at the end
  of steps T..T+3 and explodes it at the end of step T+4, where evaluate_explosions kills.

callbacks.danger_map, called on the step-T snapshot, uses "value v => deadly at the end of
step T+v", and callbacks/hunt_ceiling.escapable indexes depth so that a node at depth d is
the position at the end of step T+d-1 (start node depth 0 = position *before* the step-T move).

hunt_ceiling_v2.trap_sites(simultaneous=True) calls escapable() on `target_cells`, i.e. on the
target's position at the END of step T -- but passes the unshifted step-T danger map. The start
node is therefore labelled depth 0 ("before the step-T move") while it actually *is* the position
after that move. The target is handed one free extra move.

Correct version ("simfix"): shift the danger map one step forward, danger' = max(v-1, 0) for
v < SAFE, SAFE unchanged. Then a node at depth d from `c` is the position at the end of step T+d
and the fatality test lines up.
"simfix2" additionally drops destination cells that are already lethal at the end of step T
(danger == 0): stepping there kills the target, so it is not an escape.

Drives the world with the k=4 sim policy (the E40 primary arm) so the state distribution is the
one E40 measured on, and counts, per step where the policy actually searches for a trap, whether
each model finds any site within k and whether it finds one at distance 0 (= would BOMB now).
"""
from __future__ import annotations
import os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s
from environment import BombeRLeWorld, WorldArgs
import agent_code.benedict_task4.callbacks as cb
from hunt_ceiling_v2 import HuntCeiling, dist_map, escapable, target_cells

SAFE = cb.SAFE
K = 4
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEED = 990731


def shift(danger: np.ndarray) -> np.ndarray:
    """Advance the danger map by one step: v -> v-1, floored at 0, SAFE preserved."""
    out = danger.copy()
    m = out < SAFE
    out[m] = np.maximum(out[m] - 1, 0)
    return out


def sites(p, field, danger, bombs, model):
    """All trap sites for `p` under one of the four models."""
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
            blk = bombs | {(bx, by)}
            if model == "stale":
                cells, h = [p], hyp
            elif model == "sim":
                cells, h = target_cells(p, field, blk), hyp
            elif model == "simfix":
                cells, h = target_cells(p, field, blk), shift(hyp)
            else:  # simfix2
                cells = [c for c in target_cells(p, field, blk)
                         if c == p or danger[c] > 0]
                h = shift(hyp)
            if any(escapable(cx, cy, field, h, blk) for cx, cy in cells):
                continue
            if not escapable(bx, by, field, hyp, bombs):
                continue
            out.append((bx, by))
    return out


MODELS = ("stale", "sim", "simfix", "simfix2")
c: Counter = Counter()

q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = HuntCeiling(q, K, simultaneous=True)      # drives with the E40 primary arm

log_dir = REPO / "logs" / "audit11"
log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir),
                 save_stats=False, match_name="audit11_offbyone", seed=SEED,
                 silence_errors=False, scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None

t0 = time.time()
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
                c["steps"] += 1
                x, y = gs["self"][3]
                field = gs["field"]
                danger = cb.danger_map(gs)
                others = [o[3] for o in gs["others"]]
                if danger[x, y] >= SAFE and gs["self"][2] and others:
                    bombs = {p for p, _ in gs["bombs"]}
                    dm = dist_map(x, y, field, bombs)
                    reach = [p for p in others if dm[p] >= 0 and dm[p] <= K + s.BOMB_POWER]
                    if reach:
                        c["searched"] += 1
                        for model in MODELS:
                            best = None
                            for p in reach:
                                for site in sites(p, field, danger, bombs, model):
                                    d = int(dm[site])
                                    if d < 0 or d > K:
                                        continue
                                    if best is None or d < best:
                                        best = d
                            if best is not None:
                                c[f"fire_{model}"] += 1
                                if best == 0:
                                    c[f"bomb_{model}"] += 1
                action = policy.act(gs)
        world.do_step(action)
    if (r + 1) % 25 == 0:
        print(f"  {r+1}/{ROUNDS}  {time.time()-t0:.0f}s", flush=True)
world.end()

print(f"\nrounds={ROUNDS}  policy=k4 sim  steps={c['steps']}  "
      f"search-eligible steps={c['searched']}")
print(f"{'model':10s} {'fires':>7s} {'%steps':>8s} {'BOMB now':>9s} {'bombs/round':>12s} "
      f"{'vs sim':>8s}")
for model in MODELS:
    f, b = c[f"fire_{model}"], c[f"bomb_{model}"]
    print(f"{model:10s} {f:7d} {f/max(c['steps'],1):8.2%} {b:9d} {b/ROUNDS:12.3f} "
          f"{b/max(c['bomb_sim'],1):8.2f}x")

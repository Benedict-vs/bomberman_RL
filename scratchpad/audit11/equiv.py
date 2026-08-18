"""Audit 11 F0: is the "simultaneous-move" correction vacuous?

Claim under test: `escapable(p, hyp)` -- the E39 ("stale") test -- ALREADY enumerates the
target's step-T move. Its BFS expands p's free neighbours as depth-1 nodes under the rule
danger[n] > 0 (so a tile that burns at the end of step T is refused), and a depth-d node is the
position at the end of step T+d-1. So "the target moves to c and then escapes" is exactly the
path p -> c -> ... that the BFS from p already walks.

If that is right then the correct simultaneous-move test -- escape-less from every cell the
target can *survivably* occupy after the step, with the clock advanced by one -- must return
EXACTLY the stale site set. Five models, per searched step, full site sets compared:

  stale    E39:              cells = [p],                     danger = hyp
  sim      E40 "corrected":  cells = target_cells(p),         danger = hyp     (clock NOT advanced)
  simfix                     cells = target_cells(p),         danger = hyp-1   (clock advanced)
  simfilt                    cells = survivable target_cells, danger = hyp
  simfix2  correct           cells = survivable target_cells, danger = hyp-1
"""
from __future__ import annotations
import os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))
import settings as s
from environment import BombeRLeWorld, WorldArgs
import agent_code.benedict_task4.callbacks as cb
from hunt_ceiling_v2 import HuntCeiling, dist_map, escapable, target_cells

SAFE, K, SEED = cb.SAFE, 4, 990731
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MODELS = ("stale", "sim", "simfix", "simfilt", "simfix2")


def shift(d):
    o = d.copy(); m = o < SAFE; o[m] = np.maximum(o[m] - 1, 0); return o


def all_sites(p, field, danger, bombs, model):
    out = []
    px, py = p
    for bx in range(max(1, px - s.BOMB_POWER), min(field.shape[0], px + s.BOMB_POWER + 1)):
        for by in range(max(1, py - s.BOMB_POWER), min(field.shape[1], py + s.BOMB_POWER + 1)):
            if abs(bx - px) + abs(by - py) > s.BOMB_POWER: continue
            if field[bx, by] != 0 or (bx, by) in bombs: continue
            blast = cb.blast_coords(bx, by, field)
            if p not in blast: continue
            hyp = danger.copy()
            for (cx, cy) in blast:
                if s.BOMB_TIMER < hyp[cx, cy]: hyp[cx, cy] = s.BOMB_TIMER
            blk = bombs | {(bx, by)}
            tc = target_cells(p, field, blk)
            surv = [q for q in tc if q == p or danger[q] > 0]
            cells, h = {"stale": ([p], hyp), "sim": (tc, hyp), "simfix": (tc, shift(hyp)),
                        "simfilt": (surv, hyp), "simfix2": (surv, shift(hyp))}[model]
            if any(escapable(cx, cy, field, h, blk) for cx, cy in cells): continue
            if not escapable(bx, by, field, hyp, bombs): continue
            out.append((bx, by))
    return out


c: Counter = Counter()
q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = HuntCeiling(q, K, simultaneous=True)
log_dir = REPO / "logs" / "audit11"; log_dir.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                 match_name="a11equiv", seed=SEED, silence_errors=False, scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None
t0 = time.time()
for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r); np.random.seed(SEED + r)
    world.new_round()
    while world.running:
        alive = [a for a in world.active_agents if a.code_name == "user_agent"]
        action = "WAIT"
        if alive:
            gs = world.get_state_for_agent(alive[0])
            if gs is not None:
                c["steps"] += 1
                x, y = gs["self"][3]; field = gs["field"]; danger = cb.danger_map(gs)
                others = [o[3] for o in gs["others"]]
                if danger[x, y] >= SAFE and gs["self"][2] and others:
                    bombs = {p for p, _ in gs["bombs"]}
                    dm = dist_map(x, y, field, bombs)
                    reach = [p for p in others if dm[p] >= 0 and dm[p] <= K + s.BOMB_POWER]
                    if reach:
                        c["searched"] += 1
                        S = {}
                        for model in MODELS:
                            got = set()
                            for p in reach:
                                for site in all_sites(p, field, danger, bombs, model):
                                    if 0 <= dm[site] <= K: got.add((p, site))
                            S[model] = got
                            if got: c[f"fire_{model}"] += 1
                            if any(dm[site] == 0 for _, site in got): c[f"bomb_{model}"] += 1
                            c[f"nsites_{model}"] += len(got)
                        if S["stale"] != S["simfix2"]:
                            c["setdiff_stale_vs_correct"] += 1
                            c["only_stale"] += len(S["stale"] - S["simfix2"])
                            c["only_correct"] += len(S["simfix2"] - S["stale"])
                        if not S["sim"] <= S["stale"]: c["sim_not_subset_of_stale"] += 1
                action = policy.act(gs)
        world.do_step(action)
    if (r + 1) % 50 == 0: print(f"  {r+1}/{ROUNDS}  {time.time()-t0:.0f}s", flush=True)
world.end()

print(f"\nrounds={ROUNDS} steps={c['steps']} searched={c['searched']}")
print(f"{'model':9s} {'steps w/ site':>13s} {'%steps':>8s} {'steps w/ d=0':>13s} "
      f"{'BOMBs/round':>12s} {'total (tgt,site) pairs':>23s}")
for m in MODELS:
    print(f"{m:9s} {c['fire_'+m]:13d} {c['fire_'+m]/c['steps']:8.2%} {c['bomb_'+m]:13d} "
          f"{c['bomb_'+m]/ROUNDS:12.3f} {c['nsites_'+m]:23d}")
print(f"\nsearched steps where stale != correct(simfix2): {c['setdiff_stale_vs_correct']} "
      f"of {c['searched']}")
print(f"  pairs only in stale  : {c['only_stale']}")
print(f"  pairs only in correct: {c['only_correct']}")
print(f"searched steps where sim is NOT a subset of stale: {c['sim_not_subset_of_stale']}")

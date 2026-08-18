"""Audit 11 -- how much of "hunt the opponent" does the oracle refuse to play?

E40 fires only on *guaranteed* traps: a site counts only if EVERY cell the target
can occupy when the step resolves is escape-less. A hunting feature would not be
so restricted -- it would take a shot that kills unless the target guesses right.
This measures the size of the gap, on the control policy's state distribution so
no arm's own behaviour is baked in.

For every site that is a trap under the E39/stale rule (the target's current tile
is doomed) and that we could survive, record how many of the target's post-move
cells are doomed. `doomed == n_cells` is E40's `sim`; `doomed >= 1` is `stale`.
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

K = 4; ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300; SEED = 990731

def classify(p, field, danger, bombs):
    """-> list of (site, n_cells, n_doomed) over stale-traps we could survive."""
    out = []; px, py = p
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
            if escapable(px, py, field, hyp, blk):
                continue                       # not even a stale trap
            if not escapable(bx, by, field, hyp, bombs):
                continue                       # we die setting it
            cells = target_cells(p, field, blk)
            doomed = sum(0 if escapable(cx, cy, field, hyp, blk) else 1 for cx, cy in cells)
            out.append(((bx, by), len(cells), doomed))
    return out

q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
policy = HuntCeiling(q, -1)
ld = REPO / "logs" / "audit11"; ld.mkdir(parents=True, exist_ok=True)
args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir=str(ld), save_stats=False,
                 match_name="audit11_gap", seed=SEED, silence_errors=False, scenario="classic")
world = BombeRLeWorld(args, [("user_agent", False)] + [("rule_based_agent", False)] * 3)
world.user_input = None
c = Counter(); ties = 0; steps = 0
for r in range(ROUNDS):
    world.rng = np.random.default_rng(SEED + r); np.random.seed(SEED + r); world.new_round()
    while world.running:
        alive = [a for a in world.active_agents if a.code_name == "user_agent"]
        action = "WAIT"
        if alive:
            gs = world.get_state_for_agent(alive[0])
            if gs is not None:
                action = policy.act(gs); steps += 1
                row = q[cb.state_to_features(gs)]
                if int((row >= row.max()).sum()) > 1: ties += 1
                x, y = gs["self"][3]; field = gs["field"]; danger = cb.danger_map(gs)
                if danger[x, y] >= cb.SAFE and gs["self"][2] and gs["others"]:
                    bombs = {p for p, _ in gs["bombs"]}
                    dm = dist_map(x, y, field, bombs)
                    seen_any = {}
                    for p in (o[3] for o in gs["others"]):
                        if dm[p] < 0 or dm[p] > K + s.BOMB_POWER: continue
                        for site, ncell, nd in classify(p, field, danger, bombs):
                            d = int(dm[site])
                            if d < 0 or d > K: continue
                            key = (site, p)
                            if key in seen_any: continue
                            seen_any[key] = True
                            c[f"site_d{min(d,4)}_{nd}of{ncell}"] += 1
                            c["sites_total"] += 1
                            c["sites_guaranteed"] += (nd == ncell)
                            c["ev_sum"] += 0  # placeholder
                            if d == 0:
                                c["at0_total"] += 1
                                c["at0_guaranteed"] += (nd == ncell)
                                c["at0_ncells"] += ncell
                                c["at0_doomed"] += nd
                                c[f"at0_{nd}of{ncell}"] += 1
        world.do_step(action)
    if (r + 1) % 50 == 0: print(f"  {r+1}/{ROUNDS}", flush=True)
world.end()
print(f"\ncontrol rollout, {ROUNDS} rounds, {steps} steps, exact Q-ties on {ties} steps ({ties/steps:.4%})")
print(f"stale trap sites within k=4 (per round)      : {c['sites_total']/ROUNDS:.3f}")
print(f"  ... of which guaranteed (E40 'sim')        : {c['sites_guaranteed']/ROUNDS:.3f}"
      f"  ({c['sites_guaranteed']/max(c['sites_total'],1):.1%})")
print(f"stale trap sites we are STANDING ON (d = 0)  : {c['at0_total']/ROUNDS:.3f}")
print(f"  ... guaranteed                             : {c['at0_guaranteed']/ROUNDS:.3f}"
      f"  ({c['at0_guaranteed']/max(c['at0_total'],1):.1%})")
if c['at0_ncells']:
    print(f"  mean doomed fraction of the target's post-move cells at d=0: "
          f"{c['at0_doomed']/c['at0_ncells']:.1%}")
print("\nbreakdown at d = 0 (doomed of cells -> count):")
for k_ in sorted(k for k in c if k.startswith("at0_") and "of" in k):
    print(f"  {k_.replace('at0_',''):>8} {c[k_]:>6}   {c[k_]/ROUNDS:.4f}/round")

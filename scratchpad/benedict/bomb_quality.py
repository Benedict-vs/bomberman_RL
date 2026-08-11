"""How good are the shipped agent's bomb placements, and what was available? READ-ONLY."""
import os, sys, collections
sys.path.insert(0, "/Users/benedictvonschubert/Projects/bomberman_RL")
os.environ.setdefault("BM_QUIET_LOGS", "1")
import numpy as np
from environment import BombeRLeWorld, WorldArgs
from agent_code.benedict_task2.callbacks import blast_coords, DELTAS

def hits(x, y, field):
    return sum(1 for cx, cy in blast_coords(x, y, field) if field[cx, cy] == 1)

def best_within(x, y, field, k):
    """Max crates hittable from any free tile reachable in <= k steps."""
    seen = {(x, y)}; frontier = [(x, y)]; best = hits(x, y, field)
    for _ in range(k):
        nxt = []
        for cx, cy in frontier:
            for dx, dy in DELTAS:
                n = (cx + dx, cy + dy)
                if n in seen or field[n] != 0: continue
                seen.add(n); nxt.append(n); best = max(best, hits(n[0], n[1], field))
        frontier = nxt
    return best

args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                 save_replay=False, replay=None, make_video=False,
                 continue_without_training=True, log_dir="logs", save_stats=False,
                 match_name="bq", seed=None, silence_errors=False, scenario="classic")

actual, av1, av2, totals, destroyed = [], [], [], [], []
for r in range(40):
    w = BombeRLeWorld(args._replace(seed=990731 + r), [("benedict_task2", False)])
    w.new_round(); w.user_input = None
    a = w.agents[0]
    totals.append(int((w.arena == 1).sum()))
    while w.running:
        if not a.dead:
            x, y, f = a.x, a.y, w.arena
            before = len(w.bombs)
            w.do_step("WAIT")
            if len(w.bombs) > before:            # it dropped one
                actual.append(hits(x, y, f))
                av1.append(best_within(x, y, f, 1))
                av2.append(best_within(x, y, f, 2))
        else:
            w.do_step("WAIT")
    destroyed.append(totals[-1] - int((w.arena == 1).sum()))

a_, b_, c_ = np.array(actual), np.array(av1), np.array(av2)
print(f"bombs placed: {len(a_)} over 40 rounds ({len(a_)/40:.1f}/round)\n")
print(f"  crates hit by the bomb actually dropped : mean {a_.mean():.2f}   "
      f"distribution {dict(sorted(collections.Counter(a_).items()))}")
print(f"  best available within 1 step            : mean {b_.mean():.2f}   "
      f"(headroom {b_.mean()-a_.mean():+.2f})")
print(f"  best available within 2 steps           : mean {c_.mean():.2f}   "
      f"(headroom {c_.mean()-a_.mean():+.2f})")
print(f"  bombs already at the local optimum      : {100*(a_>=b_).mean():.0f} % (1 step) · "
      f"{100*(a_>=c_).mean():.0f} % (2 steps)")
print(f"\n  crates on the board  : {np.mean(totals):.1f} per arena")
print(f"  crates destroyed     : {np.mean(destroyed):.1f}  ->  {100*np.mean(destroyed)/np.mean(totals):.1f} % of what exists")

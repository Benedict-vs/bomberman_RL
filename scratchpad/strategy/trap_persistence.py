#!/usr/bin/env python3
"""If we walked to the trap, would it still be a trap when we got there?

`hunt_window.py` found Benedict's mechanism is real -- a trappable opponent
exists on 27 % of steps at 30-59 crates against 16.7 % on the stripped board --
but that the *reachable* supply is flat, because opponents are far away while the
crates are up (nearest 9.8 tiles vs 4.7). A trap site 6 moves away is only worth
anything if it is still a trap 6 moves later.

At every step this records the nearest reachable opponent, the closest tile from
which a bomb would leave it no escape, and our walking distance w to that tile.
It then schedules a re-check at step t + w: is a bomb at that same tile still
lethal to that same opponent? We do not actually walk there -- the agent keeps
following its own policy -- so the opponent moves unconditioned on our approach.
Against a fleeing rule_based agent that is optimistic; against one that would
have blocked our path it is pessimistic. Stated, not corrected for.

    uv run python scratchpad/strategy/trap_persistence.py --rounds 40
"""
from __future__ import annotations

import argparse, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
US = "benedict_task4"
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def dist_map(sx, sy, field, blocked):
    d = np.full(field.shape, -1, dtype=np.int16)
    d[sx, sy] = 0; q = [(sx, sy)]; head = 0
    while head < len(q):
        cx, cy = q[head]; head += 1
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if d[nx, ny] != -1 or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            d[nx, ny] = d[cx, cy] + 1; q.append((nx, ny))
    return d


def escapable(sx, sy, field, danger, blocked, horizon):
    if danger[sx, sy] >= horizon:
        return True
    q = [((sx, sy), 0)]; seen = {(sx, sy)}; head = 0
    while head < len(q):
        (cx, cy), depth = q[head]; head += 1
        if depth >= horizon:
            continue
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= horizon:
                return True
            seen.add((nx, ny)); q.append(((nx, ny), depth + 1))
    return False


def traps_for(p, field, base, bombs, cb, s, SAFE, need_self_escape):
    """Free tiles whose bomb would leave the agent at p with no escape."""
    out = []
    for bx in range(field.shape[0]):
        for by in range(field.shape[1]):
            if field[bx, by] != 0 or (bx, by) in bombs:
                continue
            if abs(bx - p[0]) + abs(by - p[1]) > s.BOMB_POWER:
                continue
            blast = cb.blast_coords(bx, by, field)
            if p not in blast:
                continue
            hyp = base.copy()
            for (cx2, cy2) in blast:
                hyp[cx2, cy2] = min(hyp[cx2, cy2], s.BOMB_TIMER)
            if escapable(p[0], p[1], field, hyp, bombs | {(bx, by)}, SAFE):
                continue
            if need_self_escape and not escapable(bx, by, field, hyp, bombs, SAFE):
                continue        # a trap we could not survive setting
            out.append((bx, by))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()
    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame                 # noqa: F401
    import agent_code.benedict_task4.callbacks as cb
    import settings as s
    SAFE = cb.SAFE

    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "persist", "seed": a.seed,
        "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
    world.user_input = None

    hit = Counter(); tot = Counter()          # keyed by walk-distance bucket
    hit_cr = Counter(); tot_cr = Counter()    # keyed by crate bucket
    walk_hist = []
    n_steps = 0

    def wb(w):
        return "0" if w == 0 else "1-2" if w <= 2 else "3-4" if w <= 4 else \
               "5-8" if w <= 8 else "9-14" if w <= 14 else "15+"

    def crb(n):
        return "30+" if n >= 30 else "10-29" if n >= 10 else "1-9" if n >= 1 else "0"

    for _ in range(a.rounds):
        world.new_round()
        pending = []          # (due_step, site, opponent name, walk, crate bucket)
        while world.running and world.step < 400:
            me = [ag for ag in world.active_agents if ag.code_name == US]
            gs = world.get_state_for_agent(me[0]) if me else None
            if gs is not None:
                n_steps += 1
                x, y = gs["self"][3]
                field = gs["field"]; base = cb.danger_map(gs)
                bombs = {p for p, _ in gs["bombs"]}
                others = {o[0]: o[3] for o in gs["others"]}
                ncr = int((field == 1).sum())

                # resolve anything due now
                still = []
                for due, site, name, w, cbk in pending:
                    if due > world.step:
                        still.append((due, site, name, w, cbk)); continue
                    tot[wb(w)] += 1; tot_cr[cbk] += 1
                    p = others.get(name)
                    if p is not None and field[site] == 0 and site not in bombs:
                        blast = cb.blast_coords(*site, field)
                        if p in blast:
                            hyp = base.copy()
                            for (cx2, cy2) in blast:
                                hyp[cx2, cy2] = min(hyp[cx2, cy2], s.BOMB_TIMER)
                            if not escapable(p[0], p[1], field, hyp,
                                             bombs | {site}, SAFE):
                                hit[wb(w)] += 1; hit_cr[cbk] += 1
                pending = still

                if others:
                    dm = dist_map(x, y, field, bombs)
                    cand = [(dm[p], n, p) for n, p in others.items() if dm[p] >= 0]
                    if cand:
                        _, name, p = min(cand)
                        sites = traps_for(p, field, base, bombs, cb, s, SAFE, True)
                        reach = [(dm[t], t) for t in sites if dm[t] >= 0]
                        if reach:
                            w, site = min(reach)
                            walk_hist.append(w)
                            pending.append((world.step + int(w), site, name,
                                            int(w), crb(ncr)))
            world.do_step("WAIT")
    world.end()

    print(f"rounds {a.rounds}, our alive steps {n_steps}")
    print(f"steps offering a self-survivable trap site somewhere: "
          f"{len(walk_hist)} ({len(walk_hist)/n_steps:.2%}), "
          f"median walk {np.median(walk_hist) if walk_hist else float('nan'):.0f}\n")
    print("== is the trap still a trap when we would arrive? ==")
    print(f"{'walk':>6}{'n':>7}{'still lethal':>14}")
    for k in ("0", "1-2", "3-4", "5-8", "9-14", "15+"):
        if tot[k]:
            print(f"{k:>6}{tot[k]:>7}{hit[k]/tot[k]:>13.1%}")
    print(f"{'ALL':>6}{sum(tot.values()):>7}"
          f"{sum(hit.values())/max(sum(tot.values()),1):>13.1%}")
    print("\n== same, by crates still standing when the plan was made ==")
    print(f"{'crates':>7}{'n':>7}{'still lethal':>14}")
    for k in ("30+", "10-29", "1-9", "0"):
        if tot_cr[k]:
            print(f"{k:>7}{tot_cr[k]:>7}{hit_cr[k]/tot_cr[k]:>13.1%}")


if __name__ == "__main__":
    main()

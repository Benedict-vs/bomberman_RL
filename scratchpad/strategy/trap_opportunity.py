#!/usr/bin/env python3
"""How often could we actually kill? The trapped-opponent opportunity rate.

E35's diagnosis was representational: digit 7 is one bit shared between "a bomb
here opens a crate" and "a bomb here catches an opponent". Before costing a
split, measure whether the second half is worth anything -- BOMB_TIMER = 4 gives
an opponent four moves to walk out of a blast, so "opponent in blast range" and
"opponent dies" are very different events.

Per greedy step, with a bomb available, this asks of the hypothetical bomb at
our own tile:
    IN_RANGE   some opponent stands in its blast
    TRAPPED    some opponent in its blast has no survivable escape, evaluated
               with the same time-aware BFS the agent uses on itself
    SAFE_FOR_US we ourselves have a survivable escape from it
and records what the agent actually did in each case.

Opponent movement is modelled as free (they may pass through each other and
through us), which makes TRAPPED an *upper* bound on real kill chances -- it
also ignores that they can drop their own bomb. Blocking tiles are walls,
crates, and live bombs.

    uv run python scratchpad/strategy/trap_opportunity.py --rounds 100
"""
from __future__ import annotations

import argparse, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
US = "benedict_task4"


def escapable(sx, sy, field, danger, blocked, horizon):
    """Can an agent at (sx, sy) reach a tile no blast covers, in time?

    Same rule as `callbacks.escape_direction`: standing on a tile at the end of
    step `depth` is fatal if the blast reaches it at or before `depth`.
    """
    if danger[sx, sy] >= horizon:            # already outside every blast
        return True
    queue = [((sx, sy), 0)]
    seen = {(sx, sy)}
    head = 0
    while head < len(queue):
        (cx, cy), depth = queue[head]; head += 1
        if depth >= horizon:
            continue
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= horizon:
                return True
            seen.add((nx, ny)); queue.append(((nx, ny), depth + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()
    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame                 # noqa: F401
    import agent_code.benedict_task4.callbacks as cb
    import settings as s

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "trap", "seed": a.seed,
        "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [(US, False)] + [("rule_based_agent", False)] * 3)
    world.user_input = None

    SAFE = cb.SAFE
    c = Counter()
    trapped_by_step = Counter()
    d7_when_trapped = Counter()
    action_when = {"trapped_safe": Counter(), "inrange_notrapped": Counter(),
                   "crate_only": Counter()}
    per_round_trapped, per_round_trapped_safe = [], []

    for _ in range(a.rounds):
        world.new_round()
        rt = rts = 0
        while world.running and world.step < 400:
            for ag in world.active_agents:
                if ag.code_name != US:
                    continue
                gs = world.get_state_for_agent(ag)
                if gs is None:
                    continue
                c["steps"] += 1
                x, y = gs["self"][3]
                if not gs["self"][2]:
                    c["no_bomb"] += 1
                    continue
                c["armed"] += 1
                field = gs["field"]
                base = cb.danger_map(gs)
                # hypothetical bomb here: it explodes at the end of step +4
                hyp = base.copy()
                for (bx, by) in cb.blast_coords(x, y, field):
                    hyp[bx, by] = min(hyp[bx, by], s.BOMB_TIMER)
                blocked = {p for p, _ in gs["bombs"]} | {(x, y)}
                others = [o[3] for o in gs["others"]]
                in_range = [p for p in others if hyp[p] < SAFE and base[p] >= SAFE
                            or (p in cb.blast_coords(x, y, field))]
                in_range = [p for p in others if p in set(cb.blast_coords(x, y, field))]
                trapped = [p for p in in_range
                           if not escapable(p[0], p[1], field, hyp, blocked, SAFE)]
                us_ok = escapable(x, y, field, hyp, {p for p, _ in gs["bombs"]}, SAFE)
                crate = any(field[cx, cy] == 1 for cx, cy in cb.blast_coords(x, y, field))
                idx = cb.state_to_features(gs)
                act = cb.ACTIONS[int(np.argmax(q[idx]))]
                if in_range: c["in_range"] += 1
                if trapped:
                    c["trapped"] += 1; rt += 1
                    d7_when_trapped[int(q[idx].argmax())] += 1
                    trapped_by_step[world.step // 50 * 50] += 1
                    if us_ok:
                        c["trapped_safe"] += 1; rts += 1
                        action_when["trapped_safe"][act] += 1
                elif in_range:
                    action_when["inrange_notrapped"][act] += 1
                elif crate and us_ok:
                    c["crate_only_safe"] += 1
                    action_when["crate_only"][act] += 1
                if crate: c["crate_in_range"] += 1
                if us_ok: c["own_escape_exists"] += 1
                if not us_ok: c["bomb_here_is_suicide"] += 1
            world.do_step("WAIT")
        per_round_trapped.append(rt); per_round_trapped_safe.append(rts)
    world.end()

    st, ar = c["steps"], c["armed"]
    print(f"rounds {a.rounds}   our alive steps {st}   armed (bomb available) "
          f"{ar} ({ar/st:.1%})")
    for k in ("in_range", "trapped", "trapped_safe", "crate_in_range",
              "crate_only_safe", "own_escape_exists", "bomb_here_is_suicide"):
        print(f"  {k:<22} {c[k]:>7}  {c[k]/st:>7.4%} of steps   "
              f"{c[k]/max(ar,1):>7.4%} of armed steps")
    print(f"\ntrapped-opportunity per round: {np.mean(per_round_trapped):.2f} steps "
          f"(and safe for us: {np.mean(per_round_trapped_safe):.2f}); "
          f"rounds with at least one: "
          f"{np.mean([t>0 for t in per_round_trapped]):.2%}")
    print(f"distinct opportunity *episodes* are fewer than steps -- a trapped "
          f"opponent stays trapped for several consecutive steps.")
    print("\nwhat the greedy policy does:")
    for k, v in action_when.items():
        tot = sum(v.values()) or 1
        print(f"  {k:<20} n={tot:>6}  " + "  ".join(
            f"{act}:{n/tot:.3f}" for act, n in v.most_common()))
    print("\ntrapped opportunities by step bucket:")
    for k in sorted(trapped_by_step):
        print(f"  step {k:>3}-{k+49:>3}: {trapped_by_step[k]}")


if __name__ == "__main__":
    main()

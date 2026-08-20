#!/usr/bin/env python3
"""E45 diagnostic -- what did the bomb that killed us actually destroy?

Benedict's observation from watching play: the agent bombs when there is nothing on
the board. `scratchpad/strategy/bomb_siting.py` measured the setup (71.5 % of armed
steps have zero crates in range, the policy bombs on 16 % of them) but never joined it
to the deaths. E43 measured the deaths (93.5 % are our own bomb) but never asked what
those bombs were worth.

This joins them. Every bomb we place is tagged at placement with what it could reach:
crates in blast, opponents in blast. When we die, the killing blast is traced back to
its bomb and we report that bomb's tag.

The number that matters: the share of our own-bomb deaths caused by a bomb that
destroyed NOTHING. That is pure downside -- it bounds what suppressing worthless bombs
could buy, with no economic cost to trade against, before any digit is designed.

Run against an EXTERNAL field as well as rule_based (E41: rule_based is unrepresentative).
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s                                              # noqa: E402
from environment import BombeRLeWorld, WorldArgs                  # noqa: E402
import agent_code.benedict_task4.callbacks as cb                  # noqa: E402
from hunt_ceiling_v2 import HuntCeiling, escapable, SAFE, DELTAS  # noqa: E402


def escape_distance(x, y, field, danger, blocked):
    """Steps to the nearest tile no blast covers, under the time-aware rule.

    Returns 99 if no escape exists. The fuse is BOMB_TIMER, so slack = BOMB_TIMER - d:
    d == BOMB_TIMER means a perfectly tight escape with no room for interference.
    """
    queue = [((x, y), 0)]; seen = {(x, y)}; head = 0
    while head < len(queue):
        (cx, cy), depth = queue[head]; head += 1
        if depth >= SAFE:
            continue
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= SAFE:
                return depth + 1
            seen.add((nx, ny)); queue.append(((nx, ny), depth + 1))
    return 99


def bomb_value(bx, by, field, others):
    """Crates and opponents the blast from (bx,by) would reach, at placement time."""
    blast = cb.blast_coords(bx, by, field)
    crates = sum(1 for (cx, cy) in blast if field[cx, cy] == 1)
    opps = sum(1 for o in others if o[3] in blast)
    return crates, opps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="ext_xiaoxiae_binary_v6")
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    policy = HuntCeiling(q, -1)          # the shipped agent exactly
    log_dir = REPO / "logs" / "e45"; log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=str(log_dir),
                     save_stats=False, match_name="e45", seed=a.seed,
                     silence_errors=False, scenario="classic")
    world = BombeRLeWorld(args, [("user_agent", False)] + [(a.field, False)] * 3)
    world.user_input = None

    c: Counter = Counter()
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)
        world.new_round()
        me = [ag for ag in world.agents if ag.code_name == "user_agent"][0]
        tags: dict = {}                      # (x,y) -> (crates, opps, placed_step)
        last_field = None
        last_pos = None
        last_step = 0
        while world.running:
            alive = [ag for ag in world.active_agents if ag.code_name == "user_agent"]
            action = "WAIT"
            if alive:
                gs = world.get_state_for_agent(alive[0])
                if gs is not None:
                    x, y = gs["self"][3]
                    last_field, last_pos, last_step = gs["field"], (x, y), gs["step"]
                    armed = gs["self"][2]
                    if armed:
                        cr, op = bomb_value(x, y, gs["field"], gs["others"])
                        c["armed_steps"] += 1
                        c[f"armed_crates_{min(cr,3)}"] += 1
                        if cr == 0:
                            c["armed_zero_crates"] += 1        # bomb_siting.py's category
                        if cr == 0 and op == 0:
                            c["armed_worthless"] += 1
                    action = policy.act(gs)
                    if action == "BOMB":
                        cr, op = bomb_value(x, y, gs["field"], gs["others"])
                        bombs_now = {q for q, _ in gs["bombs"]}
                        hyp = cb.danger_map(gs).copy()
                        for (ax, ay) in cb.blast_coords(x, y, gs["field"]):
                            if s.BOMB_TIMER < hyp[ax, ay]:
                                hyp[ax, ay] = s.BOMB_TIMER
                        d = escape_distance(x, y, gs["field"], hyp, bombs_now | {(x, y)})
                        tags[(x, y)] = (cr, op, gs["step"], d)
                        c[f"placed_esc_{min(d,5)}"] += 1
                        c["bombs_placed"] += 1
                        c[f"placed_crates_{min(cr,3)}"] += 1
                        if cr == 0 and op == 0:
                            c["placed_worthless"] += 1
                        if cr == 0:
                            c["bombed_when_zero_crates"] += 1
                        if cr == 0 and op == 0:
                            c["bombed_when_worthless"] += 1
            world.do_step(action)

        if not me.dead:
            c["survived"] += 1
            continue
        c["died"] += 1
        # which of OUR bombs' blasts covered the tile we died on?
        if last_field is None or last_pos is None:
            continue
        # A bomb that detonated long ago cannot have killed us. Without this window the
        # probe blames any bomb we ever placed whose blast geometry covers the death tile.
        # against the step WE DIED on, not world.step at round end -- the round can run for
        # hundreds of steps after our death, which made every bomb look long expired.
        fuse = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1
        killers = [p for p, (_, _, t, _d) in tags.items()
                   if last_step - t <= fuse
                   and last_pos in cb.blast_coords(p[0], p[1], last_field)]
        if not killers:
            c["death_not_own_bomb"] += 1
            continue
        c["death_own_bomb"] += 1
        worth = max(tags[p][0] for p in killers), max(tags[p][1] for p in killers)
        c[f"deathbomb_esc_{min(min(tags[p][3] for p in killers),5)}"] += 1
        c[f"deathbomb_crates_{min(worth[0],3)}"] += 1
        if worth == (0, 0):
            c["death_by_worthless_bomb"] += 1
        if (r + 1) % 100 == 0:
            print(f"  {r+1}/{a.n_rounds}", flush=True)
    world.end()

    out = REPO / "scratchpad/benedict/e45"; out.mkdir(parents=True, exist_ok=True)
    (out / f"bombvalue_{a.field}.json").write_text(json.dumps(dict(c), indent=2, sort_keys=True))

    ar, pl, di = max(c["armed_steps"],1), max(c["bombs_placed"],1), max(c["died"],1)
    ob = max(c["death_own_bomb"], 1)
    print(f"\nfield = 3x {a.field},  {a.n_rounds} rounds, seed {a.seed}")
    print(f"  armed steps                       {c['armed_steps']:6d}")
    print(f"    with 0 crates in range          {c['armed_zero_crates']:6d}  {c['armed_zero_crates']/ar:6.1%}"
          f"   (bomb_siting.py reported 71.5 %)")
    print(f"    with 0 crates AND 0 opponents   {c['armed_worthless']:6d}  {c['armed_worthless']/ar:6.1%}")
    zc = max(c['armed_zero_crates'], 1); wl = max(c['armed_worthless'], 1)
    print(f"    P(BOMB | 0 crates)              {c['bombed_when_zero_crates']/zc:6.1%}"
          f"   (bomb_siting.py reported 16 %)")
    print(f"    P(BOMB | nothing at all)        {c['bombed_when_worthless']/wl:6.1%}")
    print(f"  bombs placed                      {c['bombs_placed']:6d}   ({c['bombs_placed']/a.n_rounds:.1f}/round)")
    print(f"    placed with 0 crates 0 opps     {c['placed_worthless']:6d}  {c['placed_worthless']/pl:6.1%}  <-- pure downside")
    for k in range(4):
        lbl = f"{k}+" if k == 3 else str(k)
        print(f"    placed with {lbl} crates in blast   {c[f'placed_crates_{k}']:6d}  {c[f'placed_crates_{k}']/pl:6.1%}")
    print(f"  rounds died                       {c['died']:6d}   ({c['died']/a.n_rounds:.1%})")
    print(f"    killed by one of our own bombs  {c['death_own_bomb']:6d}  {c['death_own_bomb']/di:6.1%}")
    print(f"    ...by a bomb worth NOTHING      {c['death_by_worthless_bomb']:6d}  "
          f"{c['death_by_worthless_bomb']/ob:6.1%} of own-bomb deaths, {c['death_by_worthless_bomb']/di:6.1%} of all deaths")
    print(f"\n  => suppressing worthless bombs could address {c['death_by_worthless_bomb']/a.n_rounds:.3f} deaths/round "
          f"at zero crate cost")


if __name__ == "__main__":
    main()

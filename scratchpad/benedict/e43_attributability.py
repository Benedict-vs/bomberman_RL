#!/usr/bin/env python3
"""E43 diagnostic -- when we die, could any feature have saved us?

E42 concluded the mixed-field retrain failed because opponent-induced deaths are
"unattributable given our eight digits". That is a hypothesis about the REPRESENTATION,
and it makes a testable prediction. This probe tests it before any digit is designed.

For every death of our agent it finds `t_doom`: the last step at which some action still
survived, using only the bombs visible on the board at that moment. Then it asks what the
lethal danger was AT t_doom:

  OWN_BOMB      the doom comes from a bomb we placed  -> escape-logic failure, digits 1-5
                already carry it, and E41 showed crates/bomb is fine, so this is policy.
  VISIBLE       the killing bomb was ALREADY on the board at t_doom, and a surviving action
                existed -> the information was in `danger_map`, hence already in digits 1-5,
                and the policy simply chose a losing action. A NEW DIGIT CANNOT HELP HERE.
  UNSEEABLE     the killing bomb was placed at t_doom or later, i.e. after we committed to
                our action -> no feature over the current game_state could have shown it.
                Only a feature of the form "an opponent stands where it could bomb my escape
                route" could. THIS IS THE CASE FOR THE DIGIT.

The split VISIBLE vs UNSEEABLE is the whole point: it separates a policy problem from a
representation problem, and only the second justifies spending a digit and a sweep.

Run against an EXTERNAL field, never rule_based -- E41 showed rule_based is unrepresentative
(it suicides 1.487 of its 1.820 deaths per round, so it barely hunts).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s                                             # noqa: E402
from environment import BombeRLeWorld, WorldArgs                 # noqa: E402
import agent_code.benedict_task4.callbacks as cb                 # noqa: E402
from hunt_ceiling_v2 import HuntCeiling, escapable, SAFE         # noqa: E402

DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]      # UP RIGHT DOWN LEFT, as callbacks.py:71


def survivable_actions(x, y, field, danger, blocked):
    """Actions from (x,y) that do not lead to a state with no escape.

    Uses the same time-aware rule as callbacks.escape_direction: staying counts only if the
    tile is not already doomed, and a move counts only if its destination is enterable, not
    lethal on arrival, and escapable from there.
    """
    out = []
    if escapable(x, y, field, danger, blocked):
        out.append("WAIT")
    for (dx, dy), name in zip(DELTAS, ["UP", "RIGHT", "DOWN", "LEFT"]):
        nx, ny = x + dx, y + dy
        if field[nx, ny] != 0 or (nx, ny) in blocked:
            continue
        if danger[nx, ny] <= 0:                  # lethal the moment we arrive
            continue
        if escapable(nx, ny, field, danger, blocked):
            out.append(name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="ext_xiaoxiae_binary_v6",
                    help="opponent agent, used 3x; must NOT be rule_based_agent")
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=550731)
    ap.add_argument("--out", default="scratchpad/benedict/e43")
    a = ap.parse_args()

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    policy = HuntCeiling(q, -1)                  # k=-1: the shipped agent exactly, no override

    log_dir = REPO / "logs" / "e43"
    log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=str(log_dir), save_stats=False,
                     match_name="e43", seed=a.seed, silence_errors=False, scenario="classic")
    line_up = ["user_agent"] + [a.field] * 3
    world = BombeRLeWorld(args, [(n, False) for n in line_up])
    world.user_input = None

    c: Counter = Counter()
    doom_lead = []           # steps between t_doom and death
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)
        world.new_round()
        me = [ag for ag in world.agents if ag.code_name == "user_agent"][0]

        history = []         # per step: (step, pos, bombs_on_board, our_bombs, n_safe_actions)
        our_bombs: set = set()
        while world.running:
            alive = [ag for ag in world.active_agents if ag.code_name == "user_agent"]
            action = "WAIT"
            if alive:
                gs = world.get_state_for_agent(alive[0])
                if gs is not None:
                    x, y = gs["self"][3]
                    field = gs["field"]
                    danger = cb.danger_map(gs)
                    bombs = {p for p, _ in gs["bombs"]}
                    safe = survivable_actions(x, y, field, danger, bombs)
                    history.append((world.step, (x, y), frozenset(bombs),
                                    frozenset(our_bombs & bombs), len(safe), field))
                    action = policy.act(gs)
                    if action == "BOMB":
                        our_bombs.add((x, y))
            world.do_step(action)

        if not me.dead:
            c["survived"] += 1
            continue
        c["died"] += 1
        if not history:
            c["no_history"] += 1
            continue

        # t_doom = last recorded step at which a surviving action still existed
        doom_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i][4] > 0:
                doom_idx = i
                break
        if doom_idx is None:
            c["doomed_from_first_poll"] += 1
            continue
        doom_lead.append(history[-1][0] - history[doom_idx][0])

        step_d, pos_d, bombs_d, ours_d, nsafe_d, _ = history[doom_idx]
        # How much choice did we have at the last survivable moment? 1 = a needle the policy
        # had to find; 3+ = it simply picked a losing action with alternatives available.
        c[f"nsafe_{min(nsafe_d, 3)}"] += 1
        _, pos_f, bombs_f, ours_f, _, field_f = history[-1]

        # Which bombs actually cover the tile we died on? Anything else is irrelevant --
        # the first version of this probe classified on "do we have ANY bomb out", which is
        # nearly always true for this agent and returned 100 % OWN_BOMB.
        killers = {b for b in bombs_f if pos_f in cb.blast_coords(b[0], b[1], field_f)}
        mine = killers & set(ours_f)
        theirs = killers - mine
        visible = {b for b in theirs if b in bombs_d}      # already placed when we last had a choice
        unseen = theirs - visible                           # placed after we committed

        if not killers:
            c["other"] += 1
        elif mine and theirs:
            c["OVERLAP"] += 1        # own + enemy blast on the same tile (evaluate.py:258 blind spot)
        elif mine:
            c["OWN_BOMB"] += 1
        elif unseen:
            c["UNSEEABLE"] += 1
        else:
            c["VISIBLE"] += 1

        if (r + 1) % 50 == 0:
            print(f"  {r+1}/{a.n_rounds}", flush=True)
    world.end()

    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    died = max(c["died"], 1)
    res = {"field": a.field, "n_rounds": a.n_rounds, "seed": a.seed,
           "counts": dict(c),
           "mean_doom_lead_steps": float(np.mean(doom_lead)) if doom_lead else None,
           "shares_of_deaths": {k: c[k] / died for k in
                                ("OWN_BOMB", "OVERLAP", "VISIBLE", "UNSEEABLE",
                                 "doomed_from_first_poll", "other")}}
    (out / f"attrib_{a.field}.json").write_text(json.dumps(res, indent=2, sort_keys=True))

    print(f"\nfield = 3x {a.field},  {a.n_rounds} rounds, seed {a.seed}")
    print(f"  rounds survived            {c['survived']:5d}")
    print(f"  rounds died                {died:5d}")
    for k, label in [("OWN_BOMB",  "own bomb only   -> policy (escape logic)"),
                     ("OVERLAP",   "own + enemy blast on the same tile"),
                     ("VISIBLE",   "VISIBLE danger  -> POLICY: a new digit cannot help"),
                     ("UNSEEABLE", "UNSEEABLE       -> REPRESENTATION: the case for the digit"),
                     ("doomed_from_first_poll", "doomed at first poll"),
                     ("other",     "no bomb at t_doom")]:
        print(f"  {label:<52}{c[k]:5d}  {c[k]/died:6.1%}")
    print("  safe actions available at that last moment:")
    for k in (1, 2, 3):
        lbl = f"{k}+" if k == 3 else str(k)
        print(f"    {lbl:<50}{c[f'nsafe_{k}']:5d}  {c[f'nsafe_{k}']/died:6.1%}")
    if doom_lead:
        print(f"  mean steps between last escapable moment and death: {np.mean(doom_lead):.2f}")
    print(f"\nwrote {out / f'attrib_{a.field}.json'}")


if __name__ == "__main__":
    main()

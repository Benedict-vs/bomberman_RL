#!/usr/bin/env python3
"""Print one death case: five boards, the digits, and the exact counterfactual.

Board glyphs (printed in GUI orientation, i.e. transposed from `field[x, y]`):
  #  wall      c  crate     .  free      $  collectable coin
  @  us        1 2 3  the three rule_based agents
  b  bomb (digit = timer as the agent sees it)      *  burning explosion
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim  # noqa: E402
import analyse  # noqa: E402

NB = {0: "blocked", 1: "LETHAL", 2: "in-blast", 3: "clear"}
DIRS = {0: "NO_TARGET", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}


def render(snap) -> list[str]:
    arena = snap["arena"]
    w, h = arena.shape
    grid = [["#" if arena[x, y] == -1 else ("c" if arena[x, y] == 1 else ".")
             for x in range(w)] for y in range(h)]
    for (cx, cy) in snap["coins"]:
        grid[cy][cx] = "$"
    for coords, timer, stage, _ in snap["explosions"]:
        if stage == 0 and timer > 1:
            for (cx, cy) in coords:
                grid[cy][cx] = "*"
    for (bx, by), t, owner in snap["bombs"]:
        grid[by][bx] = str(t)
    for i, (name, (ox, oy), _) in enumerate(snap["others"]):
        grid[oy][ox] = name[-1]
    mx, my = snap["self"]
    grid[my][mx] = "@"
    return ["".join(r) for r in grid]


def digits_line(snap) -> str:
    d = snap["digits"]
    return (f"UP={NB[d[0]]:8s} RIGHT={NB[d[1]]:8s} DOWN={NB[d[2]]:8s} LEFT={NB[d[3]]:8s} | "
            f"d5 grace={d[4]}  d6={DIRS[d[5]]}  d7 bomb_useful={d[6]}  d8 dist={d[7]}")


def surviving_actions(win, dstep):
    """Which single action at the death step keeps us alive? Exact -- k = 1."""
    snap = next(s for s in win if s["step"] == dstep)
    out = []
    for a in analyse.ACTIONS:
        st = sim.state_from_snapshot(snap)
        st.do_step(sim.actions_from_snapshot(snap, a), sim.order_from_snapshot(snap))
        if st.alive("ME") and sim.survivable(st):
            out.append(a)
    return out


def show(death, case) -> None:
    win = [s for s in death["window"] if s["step"] >= death["death_step"] - 4]
    print("=" * 78)
    print(f"round {death['round']} (seed {death['seed']}), died at step "
          f"{death['death_step']}, killed by {case['killer']}, "
          f"{'own bomb' if case['own_bomb'] else 'opponent bomb'}")
    print(f"  killing bomb first visible at step {case['t_bomb']} "
          f"({case['warning']} steps of warning); "
          f"smallest surviving divergence k={case['cf_k']}")
    for s in win:
        tag = "  <-- death" if s["step"] == death["death_step"] else ""
        print(f"\n-- step {s['step']} (d{s['step'] - death['death_step']:+d}){tag}  "
              f"pos={tuple(int(v) for v in s['self'])}  bomb_left={s['bombs_left']}  "
              f"played {s['my_action']}  survivable={case['surv'][s['step']]}")
        print("   " + digits_line(s))
        print("   cand " + str(dict(zip(
            ["C1esc", "C2bombsafe", "C3oppd", "C4exits", "C5thr", "C6trap", "C7safed"],
            s["cand"]))))
        for line in render(s):
            print("      " + line)
    print(f"\n  actions at the death step that survive: {surviving_actions(win, death['death_step'])}")


def main() -> None:
    path = Path(__file__).parent / "deaths300.pkl"
    data = pickle.load(open(path, "rb"))
    cases = {c["round"]: c for c in pickle.load(open(Path(__file__).parent / "cases.pkl", "rb"))}
    wanted = [int(a) for a in sys.argv[1:]]
    for d in data["deaths"]:
        if d["round"] in wanted:
            show(d, cases[d["round"]])


if __name__ == "__main__":
    main()

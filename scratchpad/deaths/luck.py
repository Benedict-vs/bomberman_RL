#!/usr/bin/env python3
"""How much of the measured avoidability is luck the agent could not have seen?

`environment.py:poll_and_run_agents` resolves the agents in a random permutation
drawn per step, so a neighbour an opponent is standing on at the *start* of a
step can be free by the time we move into it. The counterfactual in `policy.py`
replays the recorded permutation, so a surviving action of that kind was only
available because the dice fell that way -- digits 1-4 called that tile
NB_BLOCKED and no feature the agent can compute would say otherwise.

This script splits the surviving actions at t* into ones the agent could see and
ones it could not.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import policy  # noqa: E402

DIR_INDEX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}


def main() -> None:
    data = pickle.load(open(Path(__file__).parent / "deaths300.pkl", "rb"))
    sav = {s["round"]: s for s in
           pickle.load(open(Path(__file__).parent / "savable.pkl", "rb"))}

    cats = Counter()
    for d in data["deaths"]:
        s = sav.get(d["round"])
        if s is None:
            continue
        snap = next(x for x in d["window"] if x["step"] == s["t"])
        dig = snap["digits"]
        visible = []
        for a in s["good"]:
            if a in DIR_INDEX:
                # NB_BLOCKED at the start of the step -> the agent was told this
                # move is impossible; it only worked because the opponent moved
                # first in that step's permutation.
                if dig[DIR_INDEX[a]] == 0:
                    continue
            visible.append(a)
        if not visible:
            cats["only a move into a tile digits 1-4 called BLOCKED survives"] += 1
        elif len(visible) < len(s["good"]):
            cats["a visible survivor existed, plus a lucky blocked one"] += 1
        else:
            cats["every surviving action was visible to the agent"] += 1

    n = sum(cats.values())
    print(f"# deaths with a savable step: {n}")
    for k, v in cats.most_common():
        print(f"  {v:4d} ({v/n:5.1%})  {k}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay every recorded death through `sim.py` and check it reproduces reality.

Every counterfactual in `analyse.py` rests on this simulator being an exact copy
of the engine's step. This script re-runs the recorded actions of *all* agents
from the start of each death window and asserts that our agent dies on the same
step, killed by the same bomb. Anything less and the "no action survives"
verdicts below are fiction.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim  # noqa: E402


def replay_window(death) -> tuple[int | None, set]:
    """Return (step our agent died, owners of the explosions that hit it)."""
    window = death["window"]
    st = sim.state_from_snapshot(window[0])
    for snap in window:
        st.do_step(sim.actions_from_snapshot(snap), sim.order_from_snapshot(snap))
        if not st.alive("ME"):
            owners = {ex[3] for ex in st.explosions
                      if ex[2] == 0 and (st.agents["ME"][0], st.agents["ME"][1]) in ex[0]}
            return st.step, owners
    return None, set()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "deaths300.pkl"
    data = pickle.load(open(path, "rb"))

    ok = bad_step = bad_killer = short = 0
    for d in data["deaths"]:
        if len(d["window"]) < 2:
            short += 1
            continue
        step, owners = replay_window(d)
        real_owners = {("ME" if o == "benedict_task4" else o) for o, _, _, _ in d["killers"]}
        if step != d["death_step"]:
            bad_step += 1
            if bad_step <= 5:
                print(f"  round {d['round']}: sim died at {step}, engine at {d['death_step']}")
        elif owners != real_owners:
            bad_killer += 1
            if bad_killer <= 5:
                print(f"  round {d['round']}: sim killer {owners}, engine {real_owners}")
        else:
            ok += 1

    n = len(data["deaths"])
    print(f"deaths={n}  exact={ok}  wrong_step={bad_step}  wrong_killer={bad_killer}  "
          f"window_too_short={short}")


if __name__ == "__main__":
    main()

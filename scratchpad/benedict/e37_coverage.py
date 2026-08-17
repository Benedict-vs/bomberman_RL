#!/usr/bin/env python3
"""E37 guard P3 — share of eps = 0 alive steps that land in an all-zero row.

The guard exists because E28 deferred any new digit until coverage was solved:
"every new digit makes coverage worse". E36 measured the shipped table at
0.0001 and its arms at 0.00023, so the deferral condition had expired. E37
changes what digit 8 *means* in the danger rows again, so the guard is re-run.

    BM_D8=stripe uv run python scratchpad/benedict/e37_coverage.py --arm PLB2

`BM_D8` must be exported by the caller, because `callbacks.py` reads it at
import time -- one process per arm, which is why this takes a single --arm.
`e37_coverage.sh` runs all four.

It calls the agent's own `state_to_features`; it does not reimplement it. That
is not a style preference: `scratchpad/deaths/collect.py` reimplemented it,
matched exactly until E36 changed the danger branch, and then silently produced
numbers for a feature map no agent ever used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
CKPT = REPO / "checkpoints/benedict_task4"


def rollout(arm: str, seed: int, n_rounds: int, max_steps: int = 400) -> tuple[int, int]:
    """Return (alive steps, steps whose row is all zeros) for one table."""
    suffix = f"_e37_{arm}_s{seed}__ep20000"
    # Set before the world is built: `agents.py` forks the agent as an
    # mp.Process, so it inherits this and loads the matching table.
    os.environ["BM_MODEL_SUFFIX"] = suffix
    os.environ.setdefault("BM_QUIET_LOGS", "1")

    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame                                   # noqa: F401
    import agent_code.benedict_task4.callbacks as cb

    q = np.load(CKPT / f"q_table{suffix}.npy")

    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": f"cov_{arm}_{seed}",
        "seed": 550731, "silence_errors": False, "scenario": "classic",
    })()
    agents = [("benedict_task4", False)] + [("rule_based_agent", False)] * 3
    world = BombeRLeWorld(args, agents)
    world.user_input = None                # only main.py's loop ever sets it

    alive = zero = 0
    for _ in range(n_rounds):
        world.new_round()
        while world.running and world.step < max_steps:
            for a in world.active_agents:
                if not a.name.startswith("benedict"):
                    continue
                gs = world.get_state_for_agent(a)
                if gs is None:
                    continue
                idx = cb.state_to_features(gs)
                alive += 1
                zero += int(not q[idx].any())
            world.do_step("WAIT")
    world.end()
    return alive, zero


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("ctl2", "PLB2", "PAR", "SHF"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 103, 106, 109, 112])
    ap.add_argument("--rounds", type=int, default=50)
    a = ap.parse_args()

    mode = os.environ.get("BM_D8", "")
    expect = {"ctl2": "", "PLB2": "stripe", "PAR": "parity", "SHF": "shuffle"}[a.arm]
    if mode != expect:
        raise SystemExit(f"BM_D8={mode!r} but arm {a.arm} needs {expect!r}. "
                         f"Wrong map -> wrong rows -> meaningless number.")

    tot_alive = tot_zero = 0
    for seed in a.seeds:
        alive, zero = rollout(a.arm, seed, a.rounds)
        tot_alive += alive
        tot_zero += zero
        print(f"  {a.arm} s{seed}: {zero}/{alive} = {zero / max(alive, 1):.5f}", flush=True)

    share = tot_zero / max(tot_alive, 1)
    print(f"{a.arm}  BM_D8={mode!r}  all-zero-row share {tot_zero}/{tot_alive} = "
          f"{share:.5f}   guard < 0.01 -> {'PASS' if share < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()

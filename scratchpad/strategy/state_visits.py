#!/usr/bin/env python3
"""How large is our *effective* state space at eps = 0?

64 000 rows is the storage size, not the abstraction size. Source E in
`scratchpad/survey/REPORT.md` reports 335 states; comparing that to 64 000 is a
category error unless our reachable/visited count is also 64 000. This measures:

  - distinct rows visited over N greedy rounds
  - the visit-weighted effective count (exp of the Shannon entropy of the visit
    distribution) -- the number of equally-likely rows that would produce the
    same spread, i.e. the honest "how many situations does the agent actually
    distinguish"
  - the same, split into the safe branch (digit 5 = 0) and the danger branch
  - coverage curve: how many rows cover 50/90/99/99.9 % of steps

Calls the agent's own `state_to_features`; nothing is reimplemented.

    uv run python scratchpad/strategy/state_visits.py --rounds 200
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=550731)
    ap.add_argument("--opponent", default="rule_based_agent")
    a = ap.parse_args()

    os.environ.setdefault("BM_QUIET_LOGS", "1")
    sys.path.insert(0, str(REPO))
    from environment import BombeRLeWorld
    from fallbacks import pygame                     # noqa: F401
    import agent_code.benedict_task4.callbacks as cb

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")

    args = type("A", (), {
        "no_gui": True, "fps": 15, "turn_based": False, "update_interval": 0.1,
        "save_replay": False, "replay": None, "make_video": False,
        "continue_without_training": True, "log_dir": str(REPO / "logs"),
        "save_stats": False, "match_name": "state_visits",
        "seed": a.seed, "silence_errors": False, "scenario": "classic",
    })()
    world = BombeRLeWorld(args, [("benedict_task4", False)] + [(a.opponent, False)] * 3)
    world.user_input = None

    visits: Counter[int] = Counter()
    digit_visits: list[Counter] = [Counter() for _ in range(8)]
    for _ in range(a.rounds):
        world.new_round()
        while world.running and world.step < 400:
            for ag in world.active_agents:
                if not ag.name.startswith("benedict"):
                    continue
                gs = world.get_state_for_agent(ag)
                if gs is None:
                    continue
                idx = cb.state_to_features(gs)
                visits[idx] += 1
                rem = idx
                digits = []
                for size in reversed(cb.FEATURE_SIZES):
                    digits.append(rem % size)
                    rem //= size
                for i, d in enumerate(reversed(digits)):
                    digit_visits[i][d] += 1
            world.do_step("WAIT")
    world.end()

    total = sum(visits.values())
    counts = np.array(sorted(visits.values())[::-1], dtype=float)
    p = counts / total
    eff = float(np.exp(-(p * np.log(p)).sum()))
    print(f"rounds {a.rounds}   alive steps {total}")
    print(f"distinct rows visited        {len(visits)} / {cb.N_STATES} "
          f"({len(visits)/cb.N_STATES:.4%})")
    print(f"effective rows (exp entropy) {eff:.1f}")
    cum = np.cumsum(p)
    for frac in (0.5, 0.9, 0.95, 0.99, 0.999):
        print(f"  rows covering {frac:>6.1%} of steps: {int(np.searchsorted(cum, frac)) + 1}")
    print(f"rows visited >= 10 times: {int((counts >= 10).sum())}   "
          f">= 100: {int((counts >= 100).sum())}")

    # split by branch
    def decode(idx):
        rem, ds = idx, []
        for size in reversed(cb.FEATURE_SIZES):
            ds.append(rem % size)
            rem //= size
        return list(reversed(ds))

    for label, keep in (("safe   (digit5==0)", lambda d: d[4] == 0),
                        ("danger (digit5>0) ", lambda d: d[4] > 0)):
        sub = {k: v for k, v in visits.items() if keep(decode(k))}
        st = sum(sub.values())
        c = np.array(sorted(sub.values())[::-1], dtype=float)
        pp = c / st
        print(f"{label}: {len(sub)} rows, {st} steps ({st/total:.1%}), "
              f"effective {np.exp(-(pp*np.log(pp)).sum()):.1f}")

    print("digit marginals (value: share of steps):")
    for i, c in enumerate(digit_visits):
        s = sum(c.values())
        print(f"  digit {i+1}: " + "  ".join(
            f"{v}:{n/s:.3f}" for v, n in sorted(c.items())))

    # rows with value in the shipped table
    print(f"rows with any nonzero Q in shipped table: {int((np.abs(q).sum(1) > 0).sum())}")
    unseen = [k for k in visits if not q[k].any()]
    print(f"visited rows that are all-zero in the table: {len(unseen)}")


if __name__ == "__main__":
    main()

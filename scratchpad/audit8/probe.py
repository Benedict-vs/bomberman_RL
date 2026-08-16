#!/usr/bin/env python3
"""Audit 8 probe -- what does (x+y)%4 actually carry, and did PLB use it?

Runs one trained table through the same world construction `tools/evaluate.py`
uses and instruments the agent's REAL `state_to_features` (audit 7's collect.py
defect: it reimplemented it). Records per step the eight digits, the tile, the
lattice class, and the greedy action.

Usage:
  BM_MODEL_SUFFIX=... [BM_OPPDIST_PLB=1] uv run python scratchpad/audit8/probe.py \
      --out plb.pkl --n-rounds 150
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rounds", type=int, default=150)
    ap.add_argument("--seed", type=int, default=550731)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=str(log_dir),
                     save_stats=False, match_name="a8probe", seed=a.seed,
                     silence_errors=False, scenario="classic")

    line_up = ["benedict_task4"] + ["rule_based_agent"] * 3
    world = BombeRLeWorld(args, [(name, False) for name in line_up])

    import agent_code.benedict_task4.callbacks as cb
    q = np.load(cb.MODEL_FILE)
    print("table:", cb.MODEL_FILE, "OPPDIST", cb.OPPDIST, "PLB", cb.OPPDIST_PLB,
          file=sys.stderr)

    rec = []
    st = {"round": -1}
    orig = cb.state_to_features

    def instrumented(gs: dict) -> int:
        row = orig(gs)
        x, y = gs["self"][3]
        d, r = [], row
        for size in reversed(cb.FEATURE_SIZES):
            d.append(r % size); r //= size
        d = tuple(reversed(d))
        rec.append((st["round"], int(row), d, int(x), int(y),
                    int(np.argmax(q[row])), len(gs["others"])))
        return row

    cb.state_to_features = instrumented

    rounds = []
    try:
        for ri in range(a.n_rounds):
            world.rng = np.random.default_rng(a.seed + ri)
            np.random.seed(a.seed + ri)
            st["round"] = ri
            world.new_round()
            while world.running:
                world.do_step()
            me = world.agents[0]
            rounds.append(dict(round=ri, died=int(me.dead),
                               **{k: me.statistics.get(k, 0) for k in
                                  ("score", "crates", "kills", "suicides", "steps")}))
    finally:
        world.end()

    out = Path(__file__).resolve().parent / a.out
    with out.open("wb") as fh:
        pickle.dump({"rec": rec, "rounds": rounds}, fh)
    print(f"wrote {out} steps={len(rec)} rounds={len(rounds)}", file=sys.stderr)


if __name__ == "__main__":
    main()

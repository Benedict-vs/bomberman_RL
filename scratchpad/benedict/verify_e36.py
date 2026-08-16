"""Verify the E36 feature change before spending 70 minutes on a sweep.

Three things, in order of how much they would cost to discover later:

1. **The default path is unchanged.** With both switches unset, every row index
   must equal the one the committed `callbacks.py` produces. The shipped agent
   is `q_table.npy` read through this file; if the mapping shifts, the table is
   silently being read through a different state space.
2. **The switches actually reach digit 8.** Off -> `DIST_NONE` on every danger
   step; `BM_OPPDIST=1` -> buckets in 0..3; `BM_OPPDIST_PLB=1` -> 0..3.
3. **No crash on a real round**, which is how the missing `else:` was caught --
   `target` is unbound on every *safe* step, i.e. most of them.

    uv run python scratchpad/benedict/verify_e36.py [--ref GITREF]

`--ref` is the version to compare the default path against, and it must be the
*pre-E36* one. It defaults to HEAD, which is right only while the change is
uncommitted; once E36 is committed, pass `--ref HEAD~1` or the change compares
against itself and check 1 passes vacuously.
"""

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
AGENT = "agent_code/benedict_task4/callbacks.py"


def sample_states(n_rounds: int = 3) -> list[dict]:
    """Drive the real world briefly and keep the states our agent saw."""
    sys.path.insert(0, str(REPO))
    import settings as s
    from environment import BombeRLeWorld, GenericWorld       # noqa: F401
    from fallbacks import pygame                              # noqa: F401

    args = type("A", (), {"no_gui": True, "fps": 15, "turn_based": False,
                          "update_interval": 0.1, "save_replay": False,
                          "replay": None, "make_video": False, "continue_without_training": True,
                          "log_dir": str(REPO / "logs"), "save_stats": False,
                          "match_name": "verify", "seed": 20260731, "silence_errors": False,
                          "scenario": "classic"})()
    agents = [("benedict_task4", False)] + [("rule_based_agent", False)] * 3
    world = BombeRLeWorld(args, agents)
    # `get_state_for_agent` reads it, and only `main.py`'s loop ever sets it.
    world.user_input = None
    states = []
    for _ in range(n_rounds):
        world.new_round()
        while world.running and world.step < 120:
            for a in world.active_agents:
                if a.name.startswith("benedict"):
                    states.append(world.get_state_for_agent(a))
            world.do_step("WAIT")
    world.end()
    return [st for st in states if st is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="HEAD",
                    help="git ref holding the PRE-E36 callbacks.py to compare against")
    args_cli = ap.parse_args()
    os.chdir(REPO)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and cond

    print("collecting real game states...")
    for k in ("BM_OPPDIST", "BM_OPPDIST_PLB"):
        os.environ.pop(k, None)
    states = sample_states()
    print(f"  {len(states)} states\n")

    cb = importlib.import_module("agent_code.benedict_task4.callbacks")
    rows_now = [cb.state_to_features(st) for st in states]

    # --- 1. default path identical to the committed file ---------------------
    with tempfile.TemporaryDirectory() as tmp:
        head = Path(tmp) / "callbacks_head.py"
        head.write_bytes(subprocess.check_output(
            ["git", "show", f"{args_cli.ref}:{AGENT}"]))
        spec = importlib.util.spec_from_file_location("cb_head", head)
        old = importlib.util.module_from_spec(spec)
        sys.modules["cb_head"] = old
        spec.loader.exec_module(old)
        rows_head = [old.state_to_features(st) for st in states]
    diff = sum(a != b for a, b in zip(rows_now, rows_head))
    check(diff == 0, f"default path identical to {args_cli.ref} "
                     f"({diff} of {len(states)} rows differ)")

    # --- 2. the switches reach digit 8 ---------------------------------------
    def digit8(idx):
        return idx % cb.FEATURE_SIZES[-1]

    def digit5(idx):
        base = cb.FEATURE_SIZES[-1] * cb.FEATURE_SIZES[-2] * cb.FEATURE_SIZES[-3]
        return (idx // base) % cb.FEATURE_SIZES[4]

    danger = [r for r in rows_now if digit5(r) > 0]
    check(len(danger) > 0, f"sampled {len(danger)} danger states to test against")
    check(all(digit8(r) == 0 for r in danger),
          "switches OFF: digit 8 is DIST_NONE on every danger step")

    for var, label in (("BM_OPPDIST", "BM_OPPDIST=1"), ("BM_OPPDIST_PLB", "BM_OPPDIST_PLB=1")):
        os.environ[var] = "1"
        mod = importlib.reload(cb)
        rows = [mod.state_to_features(st) for st in states]
        d = [digit8(r) for r, o in zip(rows, rows_now) if digit5(o) > 0]
        check(all(0 <= v <= 3 for v in d), f"{label}: digit 8 in 0..3 on danger steps")
        check(len(set(d)) > 1, f"{label}: digit 8 actually varies ({sorted(set(d))})")
        safe_changed = sum(1 for r, o in zip(rows, rows_now) if digit5(o) == 0 and r != o)
        check(safe_changed == 0, f"{label}: safe states are untouched")
        os.environ.pop(var)
        importlib.reload(cb)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED -- do not run the sweep"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

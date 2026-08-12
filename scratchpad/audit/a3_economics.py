"""Price the realised reward stream of a fixed table under a given reward table.

Drives BombeRLeWorld exactly as tools/evaluate.py does (per-round world.rng and
np.random reseed), lets the agent play through its own callbacks at eps=0, and
reads `me.events` after every do_step -- the same event list train.py would have
been handed.  Nothing is learnt; this only asks what the agent WOULD have been
paid.
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environment import BombeRLeWorld, WorldArgs   # noqa: E402
import events as e                                  # noqa: E402

STEP_COST = -0.1
GAMMA = 0.99

TABLES = {
    # arm A == the rung-2 table, unchanged
    "A": {e.COIN_COLLECTED: 5.0, e.CRATE_DESTROYED: 0.3, e.INVALID_ACTION: -1.0,
          e.WAITED: -0.1, e.KILLED_SELF: -5.0, e.GOT_KILLED: 0.0},
    "B": {e.COIN_COLLECTED: 5.0, e.CRATE_DESTROYED: 0.3, e.INVALID_ACTION: -1.0,
          e.WAITED: -0.1, e.KILLED_SELF: -5.0, e.GOT_KILLED: -5.0},
    # the "correct" symmetric table E25 proposes for E26
    "C": {e.COIN_COLLECTED: 5.0, e.CRATE_DESTROYED: 0.3, e.INVALID_ACTION: -1.0,
          e.WAITED: -0.1, e.KILLED_SELF: 0.0, e.GOT_KILLED: -5.0},
    # D: C with the invalid-action penalty removed (what the bomb-spam costs)
    "D": {e.COIN_COLLECTED: 5.0, e.CRATE_DESTROYED: 0.3, e.INVALID_ACTION: 0.0,
          e.WAITED: -0.1, e.KILLED_SELF: 0.0, e.GOT_KILLED: -5.0},
    # E: D plus a kill reward at the game's own 5:1 ratio
    "E": {e.COIN_COLLECTED: 5.0, e.CRATE_DESTROYED: 0.3, e.INVALID_ACTION: 0.0,
          e.WAITED: -0.1, e.KILLED_SELF: 0.0, e.GOT_KILLED: -5.0,
          e.KILLED_OPPONENT: 25.0},
}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="benedict_task3")
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=550731)
    args = ap.parse_args()

    log_dir = ROOT / "logs" / "audit_econ"
    log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit_econ", seed=None,
                      silence_errors=False, scenario="classic")
    line_up = [args.agent] + [args.opponents] * args.n_opponents
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])

    per_ep = {k: [] for k in TABLES}
    per_ep_disc = {k: [] for k in TABLES}
    terms = Counter()
    ev_tot = Counter()
    n_steps_tot = 0
    episodes = 0

    for r in range(args.n_rounds):
        world.rng = np.random.default_rng(args.seed + r)
        np.random.seed(args.seed + r)
        world.new_round()
        world.user_input = None
        me = world.agents[0]
        streams = {k: [] for k in TABLES}
        t = 0
        while world.running:
            alive_before = not me.dead
            world.do_step()
            if alive_before:
                evs = list(me.events)
                # end_round() appends SURVIVED_ROUND after the last do_step
                for k, tbl in TABLES.items():
                    streams[k].append(sum(tbl.get(v, 0.0) for v in evs) + STEP_COST)
                for v in evs:
                    ev_tot[v] += 1
                n_steps_tot += 1
                t += 1
        # SURVIVED_ROUND / final death events land in me.events after end_round
        tail = [v for v in me.events if v in (e.SURVIVED_ROUND,)]
        for v in tail:
            ev_tot[v] += 1
        for k in TABLES:
            s = np.array(streams[k])
            per_ep[k].append(s.sum())
            per_ep_disc[k].append(float((s * GAMMA ** np.arange(len(s))).sum()))
        episodes += 1

    print(f"agent table : {args.agent}  suffix={__import__('os').environ.get('BM_MODEL_SUFFIX','')}")
    print(f"field       : {args.n_opponents}x {args.opponents}, {episodes} rounds, seed {args.seed}")
    print(f"steps alive : {n_steps_tot/episodes:.1f} / episode\n")
    print("events per episode:")
    for v, c in sorted(ev_tot.items(), key=lambda kv: -kv[1]):
        print(f"  {v:<22}{c/episodes:8.3f}")
    print()
    for k in TABLES:
        u = np.array(per_ep[k]); d = np.array(per_ep_disc[k])
        print(f"reward table {k}: undiscounted return {u.mean():+8.3f} (sd {u.std():.2f})   "
              f"discounted(g=.99) {d.mean():+8.3f}")
    print(f"\nearn rate (coins+crates only, undiscounted): "
          f"{(5*ev_tot[e.COIN_COLLECTED]+0.3*ev_tot[e.CRATE_DESTROYED])/max(n_steps_tot,1):.4f} "
          f"per step, against |STEP_COST| = 0.1")


if __name__ == "__main__":
    main()

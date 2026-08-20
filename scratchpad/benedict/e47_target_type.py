#!/usr/bin/env python3
"""E47 diagnostic -- is a target-TYPE digit worth a row split?

`TASK_A_survey_vs_ours.md` section 1.1 calls this the corpus's best-evidenced feature
(K's ablation: dropping two type bits produced period-2 loops) and our largest measured
aliasing: digit 6 gives a DIRECTION but not a TYPE, and the objective is an opponent
39.1 % of safe steps, a crate 36.6 %, a coin 21.4 %.

Two things must both hold for the digit to be worth building, and neither has been checked:

  (A) The type must carry information the row does NOT already have. This is the same
      test E37 used for the lattice bit -- H(lattice | digits 1-4) was 0.192 bits of
      0.942, which is why that digit paid. Here: H(type | full row).

  (B) The aliasing must sit where the score is. callbacks.py:244-255 falls through to
      the opponent ONLY when no crate is reachable, so "opponent" is structurally the
      stripped board -- and TASK_B_argument.md section 1 measured that 94 % of the score
      already exists by step ~200. If the opponent share is all phase 2, disambiguating
      it buys nothing, and the real question shrinks to coin-vs-crate.

Ceiling-style: no training, shipped table at eps=0, run against an external field AND
rule_based (E41: rule_based is unrepresentative).
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s                                        # noqa: E402
from environment import BombeRLeWorld, WorldArgs            # noqa: E402
import agent_code.benedict_task4.callbacks as cb            # noqa: E402
from hunt_ceiling_v2 import HuntCeiling                      # noqa: E402


def target_type(x, y, field, coins, others):
    """Replicates callbacks.target_direction's fallback order and reports WHICH branch won."""
    if coins:
        cs = set(coins)
        if cb.bfs_first_step(x, y, field, lambda p: p in cs)[0] != cb.NO_TARGET:
            return "coin"
    if cb.bfs_first_step(x, y, field, lambda p: field[p] == 1)[0] != cb.NO_TARGET:
        return "crate"
    if others:
        os_ = set(others)
        if cb.bfs_first_step(x, y, field, lambda p: p in os_)[0] != cb.NO_TARGET:
            return "opponent"
    return "none"


def cond_entropy(pairs):
    """H(type | row) in bits, visit-weighted, over the rows actually seen."""
    byrow = defaultdict(Counter)
    for row, t, _ in pairs:
        byrow[row][t] += 1
    n = sum(sum(c.values()) for c in byrow.values())
    h = 0.0
    for c in byrow.values():
        m = sum(c.values())
        for k in c.values():
            p = k / m
            h -= (m / n) * p * np.log2(p)
    return h


def entropy(counter):
    n = sum(counter.values())
    return -sum((v / n) * np.log2(v / n) for v in counter.values() if v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="ext_xiaoxiae_binary_v6")
    ap.add_argument("--n-rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    policy = HuntCeiling(q, -1)
    log_dir = REPO / "logs" / "e47"; log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=str(log_dir),
                     save_stats=False, match_name="e47", seed=a.seed,
                     silence_errors=False, scenario="classic")
    world = BombeRLeWorld(args, [("user_agent", False)] + [(a.field, False)] * 3)
    world.user_input = None

    types = Counter(); pairs = []; by_phase = defaultdict(Counter)
    score_after_opp, score_total, rounds_with_opp = 0.0, 0.0, 0
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r); np.random.seed(a.seed + r)
        world.new_round()
        me = [ag for ag in world.agents if ag.code_name == "user_agent"][0]
        first_opp_score = None
        while world.running:
            alive = [ag for ag in world.active_agents if ag.code_name == "user_agent"]
            action = "WAIT"
            if alive:
                gs = world.get_state_for_agent(alive[0])
                if gs is not None:
                    x, y = gs["self"][3]
                    danger = cb.danger_map(gs)
                    if danger[x, y] >= 5:            # SAFE rows only: digit 6 is the objective
                        t = target_type(x, y, gs["field"], gs["coins"],
                                        [o[3] for o in gs["others"]])
                        types[t] += 1
                        pairs.append((cb.state_to_features(gs), t, int((gs["field"] == 1).sum())))
                        crates_left = int((gs["field"] == 1).sum())
                        bucket = ("0" if crates_left == 0 else
                                  "1-9" if crates_left < 10 else
                                  "10-29" if crates_left < 30 else
                                  "30-59" if crates_left < 60 else "60+")
                        by_phase[bucket][t] += 1
                        if t == "opponent" and first_opp_score is None:
                            first_opp_score = me.score or 0
                    action = policy.act(gs)
            world.do_step(action)
        final = me.statistics.get("score", 0)   # written by end_round, so valid here
        score_total += final
        if first_opp_score is not None:
            rounds_with_opp += 1
            score_after_opp += final - first_opp_score
        if (r + 1) % 50 == 0:
            print(f"  {r+1}/{a.n_rounds}", flush=True)
    world.end()

    n = sum(types.values())
    H = entropy(types); Hc = cond_entropy(pairs)
    # The cut that decides it: the opponent class is structurally "crates == 0", a phase
    # worth a few percent of the score. Restrict to the crate phase, where the score is,
    # and the question becomes whether COIN vs CRATE is already determined by the row.
    live = [(r, t, c) for (r, t, c) in pairs if c > 0 and t in ("coin", "crate")]
    Hl = entropy(Counter(t for _, t, _ in live)) if live else 0.0
    Hlc = cond_entropy(live) if live else 0.0
    out = {"field": a.field, "n_rounds": a.n_rounds, "safe_steps": n,
           "types": dict(types), "H_type": H, "H_type_given_row": Hc,
           "mutual_information": H - Hc,
           "crate_phase_steps": len(live),
           "H_cointype_cratephase": Hl, "H_cointype_given_row_cratephase": Hlc,
           "score_total": score_total, "score_after_first_opponent": score_after_opp,
           "rounds_reaching_opponent": rounds_with_opp,
           "by_phase": {k: dict(v) for k, v in by_phase.items()}}
    d = REPO / "scratchpad/benedict/e47"; d.mkdir(parents=True, exist_ok=True)
    (d / f"targettype_{a.field}.json").write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"\nfield = 3x {a.field}, {a.n_rounds} rounds, {n} safe steps")
    print("\n  (A) does the type carry information the row does NOT already have?")
    print(f"      H(type)            = {H:.3f} bits")
    print(f"      H(type | full row) = {Hc:.3f} bits      <-- what a type digit would ADD")
    print(f"      mutual information = {H-Hc:.3f} bits    (E37's lattice bit: 0.192 of 0.942)")
    print(f"\n  (A2) restricted to the CRATE PHASE (crates > 0), coin vs crate only")
    print(f"       {len(live)} steps ({len(live)/n:.1%} of safe steps)")
    print(f"       H(coin|crate)            = {Hl:.3f} bits")
    print(f"       H(coin|crate | full row) = {Hlc:.3f} bits   <-- what the FREE re-partition would add")
    print(f"       mutual information       = {Hl-Hlc:.3f} bits")
    print("\n  (B) where does the aliasing sit?")
    print(f"      {'crates left':<12}" + "".join(f"{t:>11}" for t in ("coin","crate","opponent","none")))
    for b in ("60+", "30-59", "10-29", "1-9", "0"):
        row = by_phase.get(b, Counter()); m = max(sum(row.values()), 1)
        print(f"      {b:<12}" + "".join(f"{row[t]/m:>10.1%} " for t in ("coin","crate","opponent","none")))
    print(f"\n      overall: " + "  ".join(f"{t} {types[t]/n:.1%}" for t in ("coin","crate","opponent","none")))
    if rounds_with_opp:
        print(f"      rounds that reach the stripped board: {rounds_with_opp}/{a.n_rounds}")
        print(f"      score earned AFTER the board strips: {score_after_opp/a.n_rounds:.3f}/round "
              f"of {score_total/a.n_rounds:.3f} total = {score_after_opp/max(score_total,1e-9):.1%} "
              f"of all score")


if __name__ == "__main__":
    main()

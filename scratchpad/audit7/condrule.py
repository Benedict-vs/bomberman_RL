#!/usr/bin/env python3
"""Audit 7: is the escape ceiling *conditional* on opponent proximity?

Audit 5 measured the ceiling by forcing the argmax to digit 6 in EVERY danger
row (17.4 % of steps): score 3.719 -> 4.399, won 0.372 -> 0.442. E33 (reward)
and E34 (initial condition) both failed to reach it, and both failed the same
way -- the escape behaviour was installed and the *bombing* degraded.

Hypothesis under test: the ceiling is not "always escape", it is "escape when an
opponent is close", and the eight digits cannot express the condition. If so a
rule restricted to the near-opponent subset should capture most of the ceiling
while firing on a small fraction of the steps, and the complementary rule
(escape only when NO opponent is close) should capture little.

Four arms, evaluation only, no training, identical arenas:
  ctl   -- the shipped table, unchanged
  esc   -- force digit 6 whenever own_danger > 0 and digit 6 != 0   (audit 5)
  near  -- force digit 6 only when additionally BFS(opponent) <= 2
  far   -- force digit 6 only when additionally BFS(opponent) >= 3  (placebo:
           the complement of `near`, same instrument, wrong condition)

Nothing outside scratchpad/audit7/ is written and no agent file is edited; the
rule is applied by wrapping `act` in this harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from environment import BombeRLeWorld, WorldArgs  # noqa: E402

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ctl", "esc", "near", "far"], required=True)
    ap.add_argument("--n-rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=990731)
    ap.add_argument("--near-max", type=int, default=2)
    args = ap.parse_args()

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    wargs = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=str(log_dir), save_stats=False,
        match_name=f"a7cond_{args.arm}", seed=args.seed, silence_errors=False,
        scenario="classic",
    )
    line_up = ["benedict_task4"] + ["rule_based_agent"] * 3
    world = BombeRLeWorld(wargs, [(name, False) for name in line_up])

    import agent_code.benedict_task4.callbacks as cb

    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    fired = {"n": 0, "steps": 0}

    def act(self, game_state: dict) -> str:
        row = cb.state_to_features(game_state)
        fired["steps"] += 1

        if args.arm != "ctl":
            field = game_state["field"]
            x, y = game_state["self"][3]
            danger = cb.danger_map(game_state)
            own_danger = 0 if danger[x, y] >= cb.SAFE else int(danger[x, y]) + 1
            if own_danger:
                occupied = {p for p, _ in game_state["bombs"]}
                occupied.update(o[3] for o in game_state["others"])
                esc = cb.escape_direction(x, y, field, danger, occupied)
                if esc != cb.NO_TARGET:
                    others = [o[3] for o in game_state["others"]]
                    if others:
                        oset = set(others)
                        _, od = cb.bfs_first_step(x, y, field, lambda p: p in oset)
                        if od == 0:
                            od = 99
                    else:
                        od = 99
                    ok = (args.arm == "esc"
                          or (args.arm == "near" and od <= args.near_max)
                          or (args.arm == "far" and od > args.near_max))
                    if ok:
                        fired["n"] += 1
                        return ACTIONS[esc - 1]

        qr = q[row]
        best = np.flatnonzero(qr >= qr.max())
        return ACTIONS[int(best[0] if best.size == 1
                           else self.policy_rng.choice(best))]

    cb.act = act

    rows = []
    try:
        for ri in range(args.n_rounds):
            world.rng = np.random.default_rng(args.seed + ri)
            np.random.seed(args.seed + ri)
            world.new_round()
            while world.running:
                world.do_step()
            me = world.agents[0]
            best = max(a.statistics.get("score", 0) for a in world.agents)
            rows.append(dict(
                round=ri,
                score=me.statistics.get("score", 0),
                won=int(me.statistics.get("score", 0) == best),
                kills=me.statistics.get("kills", 0),
                coins=me.statistics.get("coins", 0),
                crates=me.statistics.get("crates", 0),
                bombs=me.statistics.get("bombs", 0),
                suicides=me.statistics.get("suicides", 0),
                survived=int(not me.dead),
                died=int(me.dead),
            ))
            if (ri + 1) % 50 == 0:
                print(f"  {args.arm} {ri+1}/{args.n_rounds}", file=sys.stderr, flush=True)
    finally:
        world.end()

    out = Path(__file__).resolve().parent / f"cond_{args.arm}.json"
    out.write_text(json.dumps(
        {"arm": args.arm, "seed": args.seed, "n": args.n_rounds,
         "fire_rate": fired["n"] / max(fired["steps"], 1),
         "steps": fired["steps"], "rows": rows}, indent=0))
    print(f"{args.arm}: score {np.mean([r['score'] for r in rows]):.3f} "
          f"won {np.mean([r['won'] for r in rows]):.3f} "
          f"fire_rate {fired['n']/max(fired['steps'],1):.4f}")


if __name__ == "__main__":
    main()

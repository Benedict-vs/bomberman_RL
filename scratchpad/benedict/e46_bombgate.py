#!/usr/bin/env python3
"""E46 ceiling -- don't bomb without escape ROOM (as opposed to without a reason).

E45: bombs that clear nothing are the SAFE ones (0.40x death lift vs 3+ crate bombs at
1.54x), so "only bomb when needed" would delete the harmless half and keep the killers.
E43: in 23.3 % of deaths against a strong field the agent was down to ONE surviving
action, against 9.3 % vs rule_based. TASK_A section 1.2: an escape from our own bomb
exists on 99.08 % of armed steps, so the binary "is BOMB safe" bit is a constant --
but nobody has measured how MANY escapes exist.

This gates BOMB on the COUNT. When the shipped greedy policy says BOMB, count the
survivable escape routes that would remain after the bomb lands; if fewer than N,
take the best non-BOMB action instead. N=1 is the existing near-constant bit and
should reproduce the control -- it is the internal-validity check, not an arm.

An oracle, so an upper bound: it recomputes the true post-bomb escape set every step
with the same time-aware BFS callbacks.escape_direction uses. A tabular digit could
only offer a bucketed version of this.

Nothing under agent_code/ is written: the line-up drives the provided user_agent.
"""
from __future__ import annotations

import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("BM_QUIET_LOGS", "1")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad/benedict"))

import settings as s                                                      # noqa: E402
from environment import BombeRLeWorld, WorldArgs                          # noqa: E402
import agent_code.benedict_task4.callbacks as cb                          # noqa: E402
from hunt_ceiling_v2 import HuntCeiling, escapable, dist_map, SAFE, DELTAS  # noqa: E402

FRAMEWORK_STATS = ["score", "coins", "kills", "suicides", "crates", "bombs",
                   "moves", "invalid", "steps", "time"]


def escape_distance(x, y, field, danger, blocked):
    """Steps from (x,y) to the nearest tile no blast covers, time-aware. 99 if none.

    The fuse is BOMB_TIMER = 4, so slack = 4 - d. A pilot showed the ROUTE COUNT is the
    wrong variable -- requiring two post-bomb escapes vetoes 99.6 % of bombs, because a
    bomb covering your tile and all four arms usually leaves exactly one way out. The
    distance is the variable that discriminates: E45 measured d >= 4 at 2.9 % of bombs
    and 20.5 % of own-bomb deaths, a 7.1x lift.
    """
    queue = [((x, y), 0)]; seen = {(x, y)}; head = 0
    while head < len(queue):
        (cx, cy), depth = queue[head]; head += 1
        if depth >= SAFE:
            continue
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if danger[nx, ny] <= depth:
                continue
            if danger[nx, ny] >= SAFE:
                return depth + 1
            seen.add((nx, ny)); queue.append(((nx, ny), depth + 1))
    return 99


class BombGate(HuntCeiling):
    """Shipped greedy policy, except BOMB is vetoed when post-bomb escape room < min_routes."""

    def __init__(self, q, veto_at: int):
        super().__init__(q, -1)              # k=-1: no hunt override, this is the shipped policy
        self.veto_at = veto_at               # veto BOMB when escape distance >= this; 0 disables
        self.vetoed = 0
        self.bomb_chances = 0

    def act(self, game_state):
        action = self.greedy(game_state)
        if self.veto_at <= 0 or action != "BOMB":
            self.steps += 1
            return action
        self.steps += 1
        self.bomb_chances += 1
        x, y = game_state["self"][3]
        field = game_state["field"]
        bombs = {p for p, _ in game_state["bombs"]}
        hyp = cb.danger_map(game_state).copy()
        for (cx, cy) in cb.blast_coords(x, y, field):
            if s.BOMB_TIMER < hyp[cx, cy]:
                hyp[cx, cy] = s.BOMB_TIMER
        if escape_distance(x, y, field, hyp, bombs | {(x, y)}) < self.veto_at:
            return "BOMB"
        self.vetoed += 1
        # second-best action from the same row, so the veto changes only the bomb decision
        row = self.q[cb.state_to_features(game_state)].copy()
        row[cb.ACTIONS.index("BOMB")] = -np.inf
        best = np.flatnonzero(row >= row.max() - cb.TIE_TOL)
        return cb.ACTIONS[int(best[0] if best.size == 1 else self.policy_rng.choice(best))]


def commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if d else "")
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--veto-at", type=int, required=True,
                    help="veto BOMB when post-bomb escape distance >= this; 0 disables")
    ap.add_argument("--field", default="ext_xiaoxiae_binary_v6")
    ap.add_argument("--n-rounds", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=990731)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", default="scratchpad/benedict/e46")
    a = ap.parse_args()

    label = a.label or f"e46_v{a.veto_at}_{a.field}"
    q = np.load(REPO / "agent_code/benedict_task4/q_table.npy")
    policy = BombGate(q, a.veto_at)
    log_dir = REPO / "logs" / "e46"; log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                     save_replay=False, replay=None, make_video=False,
                     continue_without_training=True, log_dir=str(log_dir),
                     save_stats=False, match_name=label, seed=a.seed,
                     silence_errors=False, scenario="classic")
    line_up = ["user_agent"] + [a.field] * 3
    world = BombeRLeWorld(args, [(n, False) for n in line_up])
    world.user_input = None

    records, started = [], time.time()
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)                 # reaches the provided opponents
        world.new_round()
        while world.running:
            alive = [ag for ag in world.active_agents if ag.code_name == "user_agent"]
            action = "WAIT"
            if alive:
                gs = world.get_state_for_agent(alive[0])
                if gs is not None:
                    action = policy.act(gs)
            world.do_step(action)
        rr = []
        for slot, agent in enumerate(world.agents):
            rec = {"round": r, "seed": a.seed + r, "slot": slot, "agent": agent.name,
                   "code": agent.code_name, "survived": int(not agent.dead),
                   "round_steps": world.step}
            for k in FRAMEWORK_STATS:
                rec[k] = agent.statistics.get(k, 0)
            rec["died"] = int(agent.dead)
            rec["killed_by_opponent"] = max(0, rec["died"] - rec["suicides"])
            rr.append(rec)
        best = max(x["score"] for x in rr)
        for rec in rr:
            rec["rank"] = 1 + sum(1 for o in rr if o["score"] > rec["score"])
            rec["won"] = int(rec["score"] == best)
        records.extend(rr)
        if (r + 1) % 250 == 0:
            print(f"  {r+1}/{a.n_rounds}  {time.time()-started:.0f}s", flush=True)
    world.end()

    out = REPO / a.out_dir; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{label}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    meta = {"label": label, "veto_at": a.veto_at, "field": a.field, "seed": a.seed,
            "n_rounds": a.n_rounds, "line_up": line_up, "commit": commit(),
            "table": "agent_code/benedict_task4/q_table.npy",
            "bomb_chances": policy.bomb_chances, "vetoed": policy.vetoed,
            "policy_steps": policy.steps,
            "note": "ceiling probe, not a trained agent; user_agent driven by e46_bombgate.py"}
    (out / f"{label}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    us = [x for x in records if x["code"] == "user_agent"]
    m = lambda k: float(np.mean([x[k] for x in us]))
    byr = {}
    for rec in records: byr.setdefault(rec["round"], []).append(rec)
    mm = np.mean([next(x["score"] for x in v if x["code"] == "user_agent")
                  - np.mean([o["score"] for o in v if o["code"] != "user_agent"]) for v in byr.values()])
    print(f"\nveto_at={a.veto_at} field={a.field} n={len(us)}")
    print(f"  score {m('score'):.3f}  margin_mean {mm:+.3f}  coins {m('coins'):.3f}  "
          f"kills {m('kills'):.3f}  suicides {m('suicides'):.3f}  crates {m('crates'):.2f}  "
          f"bombs {m('bombs'):.2f}  survived {m('survived'):.3f}")
    print(f"  BOMB vetoed on {policy.vetoed}/{policy.bomb_chances} chances "
          f"({policy.vetoed/max(policy.bomb_chances,1):.1%}), {policy.vetoed/a.n_rounds:.2f}/round")


if __name__ == "__main__":
    main()

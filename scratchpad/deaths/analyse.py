#!/usr/bin/env python3
"""Death taxonomy for `benedict_task4` in a 3x `rule_based_agent` field.

Every column here is mechanical: it comes either straight out of the recorded
snapshot, or out of the simulator that `validate.py` checks against the engine.
Nothing in this file is a judgement about a board.

Definitions used throughout
---------------------------
d            step on which our agent was removed.
killer       owner of the explosion that covered its tile at step d.
t_bomb       step at which the killing bomb first appears in the window
             (None if it was already there when the window opened).
survivable(t) frozen-opponent space-time BFS: from the position at the *start*
             of step t, does any move sequence dodge every blast that is already
             on the board?  Opponents stand still and drop nothing more, so a
             False here is a hard verdict and a True is only "there was a way
             out against what it could already see".
t_doom       first step of the window from which survivable is False for every
             later step -- the last moment the death could still be dodged.
cf_k         result of the *real* counterfactual: replay the opponents' recorded
             actions and search all 6^k action sequences of ours from step d-k+1.
             cf_k = smallest k for which one of them is alive at step d and
             `survivable` afterwards. None = not avoidable within 5 steps.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sim  # noqa: E402

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
NB = {0: "blocked", 1: "lethal", 2: "in-blast", 3: "clear"}
DIRS = {0: "-", 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}


# ---------------------------------------------------------------------------
def bfs_dist(arena, start, goals: set) -> int | None:
    """Shortest walking distance to any goal tile (goals may be occupied)."""
    if not goals:
        return None
    seen = {tuple(start)}
    frontier = [tuple(start)]
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for (x, y) in frontier:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                p = (x + dx, y + dy)
                if p in seen:
                    continue
                if p in goals:
                    return depth
                if arena[p] != 0:
                    continue
                seen.add(p)
                nxt.append(p)
        frontier = nxt
        if depth > 60:
            break
    return None


def analyse_death(d: dict) -> dict | None:
    win = d["window"]
    if len(win) < 2:
        return None
    step_of = {s["step"]: s for s in win}
    dstep = d["death_step"]
    if dstep not in step_of:
        return None

    killers = d["killers"]
    owners = {o for o, _, _, _ in killers}
    own_bomb = "benedict_task4" in owners
    killer = "ME" if own_bomb else (sorted(owners)[0] if owners else "?")
    origins = {tuple(int(v) for v in orig) for _, orig, _, _ in killers}

    # -- when did the killing bomb appear -----------------------------------
    t_bomb = None
    for s in win:
        here = {tuple(int(v) for v in p) for p, _, o in s["bombs"]
                if (o in owners or not owners)}
        if here & origins:
            t_bomb = s["step"]
            break
    # It may already have exploded into an explosion before the window opened.
    warning = None if t_bomb is None else dstep - t_bomb

    # -- survivability per step ---------------------------------------------
    surv = {}
    for s in win:
        surv[s["step"]] = sim.survivable(sim.state_from_snapshot(s))
    t_doom = None
    for s in win:
        t = s["step"]
        if all(not surv[u["step"]] for u in win if u["step"] >= t):
            t_doom = t
            break

    # -- did it walk in, or was it standing there? ---------------------------
    prev = step_of.get(dstep - 1)
    last = step_of[dstep]
    moved_in = None
    if prev is not None:
        moved_in = tuple(prev["self"]) != tuple(last["self"])

    # -- the real counterfactual --------------------------------------------
    cf_k, cf_first = counterfactual(win, dstep)

    # -- context -------------------------------------------------------------
    arena = last["arena"]
    opp_tiles = {tuple(int(v) for v in o[1]) for o in last["others"]}
    opp_dist = bfs_dist(arena, tuple(last["self"]), opp_tiles)
    dig = last["digits"]

    # -- own-bomb deaths: was the drop itself already fatal? ------------------
    drop_step = drop_safe = None
    if own_bomb:
        for s in win:
            if s["my_action"] == "BOMB" and tuple(int(v) for v in s["self"]) in origins:
                drop_step = s["step"]
                drop_safe = s["cand"][1]     # C2_bomb_safe, computed before the drop
                break

    # -- opponent deaths: was the threat visible one step before the bomb? ----
    threat_seen = trap_seen = None
    if not own_bomb and t_bomb is not None:
        pre = step_of.get(t_bomb - 1)
        if pre is not None:
            threat_seen = pre["cand"][4]     # C5_enemy_threat
            trap_seen = pre["cand"][5]       # C6_trap

    return {
        "round": d["round"], "death_step": dstep, "killer": killer,
        "drop_step": drop_step, "drop_safe": drop_safe,
        "threat_seen": threat_seen, "trap_seen": trap_seen,
        "cand_at_death": last["cand"],
        "own_bomb": own_bomb, "t_bomb": t_bomb, "warning": warning,
        "surv": surv, "t_doom": t_doom, "moved_in": moved_in,
        "cf_k": cf_k, "cf_first": cf_first,
        "opp_dist": opp_dist, "digits_at_death": dig,
        "digits_seq": [(s["step"], s["digits"], s["my_action"]) for s in win],
        "score": d["score"], "crates_left": last["crates_left"],
        "my_bomb_live": any(o == "benedict_task4" for _, _, o in last["bombs"]),
        "n_others": len(last["others"]),
    }


def counterfactual(win, dstep, max_k: int = 5):
    """Smallest k such that some own action sequence from step d-k+1 survives.

    Opponents replay their recorded actions, so for k = 1 this is exact: the
    world is *identical* up to our own last move. For larger k the opponents no
    longer react to us, which makes a "survives" verdict optimistic and a
    "nothing survives" verdict conservative -- both are stated as such.
    """
    steps = [s["step"] for s in win]
    for k in range(1, max_k + 1):
        start = dstep - k + 1
        if start not in steps:
            return None, None
        idx = steps.index(start)
        seq_snaps = [s for s in win if s["step"] >= start]
        base = sim.state_from_snapshot(seq_snaps[0])
        for combo in itertools.product(ACTIONS, repeat=k):
            st = base.clone()
            dead = False
            for i, snap in enumerate(seq_snaps[:k]):
                st.do_step(sim.actions_from_snapshot(snap, combo[i]),
                           sim.order_from_snapshot(snap))
                if not st.alive("ME"):
                    dead = True
                    break
            if dead:
                continue
            if sim.survivable(st):
                return k, combo
    return None, None


# ---------------------------------------------------------------------------
def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "deaths300.pkl"
    data = pickle.load(open(path, "rb"))
    rounds = data["rounds"]
    n_rounds = data["n_rounds"]

    cases = [c for c in (analyse_death(d) for d in data["deaths"]) if c]
    print(f"# rounds={n_rounds}  deaths={len(data['deaths'])}  analysed={len(cases)}")

    died = sum(r["died"] for r in rounds)
    suic = sum(r["suicides"] for r in rounds)
    print(f"died/round={died/n_rounds:.3f}  suicides/round={suic/n_rounds:.3f}  "
          f"killed_by/round={(died-suic)/n_rounds:.3f}  "
          f"score={np.mean([r['score'] for r in rounds]):.3f}  "
          f"steps_alive={np.mean([r['alive_steps'] for r in rounds]):.1f}")

    # -- taxonomy ------------------------------------------------------------
    def bucket(c):
        if c["own_bomb"]:
            return "own bomb"
        if c["t_bomb"] is None:
            return "opp bomb, dropped before window"
        if c["warning"] <= 1:
            return "opp bomb, <=1 step warning"
        if not c["surv"].get(c["t_bomb"], True):
            return "opp bomb, trapped when it landed"
        return "opp bomb, escape existed"

    tax = Counter(bucket(c) for c in cases)
    print("\n## taxonomy")
    for k, v in tax.most_common():
        sub = [c for c in cases if bucket(c) == k]
        forfeit = np.mean([400 - c["death_step"] for c in sub])
        print(f"  {k:42s} n={v:4d} ({v/len(cases):5.1%})  "
              f"mean steps forfeited={forfeit:5.1f}  "
              f"cf_k=1 avoidable={np.mean([c['cf_k']==1 for c in sub]):.2f}  "
              f"cf unavoidable={np.mean([c['cf_k'] is None for c in sub]):.2f}")

    print("\n## counterfactual: smallest k whose 6^k own-action sequences survive")
    ck = Counter(c["cf_k"] for c in cases)
    for k in [1, 2, 3, 4, 5, None]:
        if ck[k]:
            print(f"  k={str(k):5s} n={ck[k]:4d} ({ck[k]/len(cases):5.1%})")

    print("\n## survivability at the step the killing bomb landed")
    with_t = [c for c in cases if c["t_bomb"] is not None]
    print(f"  had an escape when it landed: "
          f"{np.mean([c['surv'][c['t_bomb']] for c in with_t]):.3f}  (n={len(with_t)})")
    print("  warning (steps between bomb landing and death):",
          dict(sorted(Counter(c["warning"] for c in with_t).items())))

    print("\n## who killed us")
    print("  ", dict(Counter(c["killer"] for c in cases).most_common()))

    print("\n## digits at the death step (own_danger, escape dir)")
    for name, sub in (("own bomb", [c for c in cases if c["own_bomb"]]),
                      ("opponent", [c for c in cases if not c["own_bomb"]])):
        d5 = Counter(c["digits_at_death"][4] for c in sub)
        d6 = Counter(c["digits_at_death"][5] for c in sub)
        print(f"  {name:9s} digit5 {dict(sorted(d5.items()))}  "
              f"digit6 {dict(sorted(d6.items()))}")

    # -- the blindness test --------------------------------------------------
    print("\n## the blindness test: digits at step d-2 and d-1")
    for lag in (3, 2, 1, 0):
        rows = []
        for c in cases:
            m = {s: dg for s, dg, _ in c["digits_seq"]}
            s = c["death_step"] - lag
            if s in m:
                rows.append(m[s])
        if not rows:
            continue
        safe_looking = sum(1 for r in rows if r[4] == 0 and all(v in (0, 3) for v in r[:4]))
        print(f"  step d-{lag}: n={len(rows):4d}  "
              f"digit5==0 (tile safe): {np.mean([r[4]==0 for r in rows]):.3f}  "
              f"'nothing wrong anywhere' (d5=0 and every neighbour blocked/clear): "
              f"{safe_looking/len(rows):.3f}")

    # base rate over all alive steps
    log = data["step_log"]
    base_safe = np.mean([e["digits"][4] == 0 and all(v in (0, 3) for v in e["digits"][:4])
                         for e in log])
    print(f"  base rate over all {len(log)} alive steps: {base_safe:.3f}")

    # -- opponent proximity ---------------------------------------------------
    print("\n## opponent proximity (BFS tiles) at the death step vs. base rate")
    dists = [c["opp_dist"] for c in cases if c["opp_dist"] is not None]
    print(f"  at death: median={np.median(dists):.1f}  <=3: "
          f"{np.mean([d <= 3 for d in dists]):.3f}  n={len(dists)}")

    # -- cost ranking ---------------------------------------------------------
    print("\n## cost ranking (per 300 rounds)")
    for k, v in tax.most_common():
        sub = [c for c in cases if bucket(c) == k]
        forfeit = sum(400 - c["death_step"] for c in sub) / n_rounds
        gift = sum(0 if c["own_bomb"] else 5 for c in sub) / n_rounds
        print(f"  {k:42s} {v/n_rounds:6.3f}/round  "
              f"steps forfeited/round={forfeit:6.1f}  points gifted/round={gift:5.3f}")

    # -- own-bomb deaths: fatal drop vs botched escape ------------------------
    own = [c for c in cases if c["own_bomb"]]
    found = [c for c in own if c["drop_step"] is not None]
    print(f"\n## own-bomb deaths (n={len(own)}, drop step located for {len(found)})")
    print(f"  drop was already unsurvivable (C2_bomb_safe=0): "
          f"{sum(1 for c in found if c['drop_safe'] == 0)} / {len(found)}")
    print(f"  drop was survivable, escape then botched:       "
          f"{sum(1 for c in found if c['drop_safe'] == 1)} / {len(found)}")
    print(f"  digit6 = NO_TARGET at the death step:           "
          f"{sum(1 for c in own if c['digits_at_death'][5] == 0)} / {len(own)}")

    # digit 6 says "no escape" but there was one
    no_esc = [c for c in cases if c["digits_at_death"][5] == 0
              and c["digits_at_death"][4] > 0]
    print(f"\n## digit 6 = NO_TARGET while digit 5 > 0 (agent's escape BFS gave up)")
    print(f"  n={len(no_esc)}  of which a move sequence did survive (cf_k=1): "
          f"{sum(1 for c in no_esc if c['cf_k'] == 1)}")

    # -- opponent deaths: threat visible before the bomb landed? --------------
    opp = [c for c in cases if not c["own_bomb"] and c["threat_seen"] is not None]
    print(f"\n## opponent-bomb deaths, state one step BEFORE the bomb landed (n={len(opp)})")
    print(f"  C5_enemy_threat=1 (an armed opponent already covered my tile): "
          f"{sum(1 for c in opp if c['threat_seen'] == 1)} / {len(opp)}")
    print(f"  C6_trap=1 (…and that bomb would have left me no way out):      "
          f"{sum(1 for c in opp if c['trap_seen'] == 1)} / {len(opp)}")

    with open(Path(__file__).parent / "cases.pkl", "wb") as f:
        pickle.dump(cases, f)
    print(f"\nwrote cases.pkl ({len(cases)} cases)")


if __name__ == "__main__":
    main()

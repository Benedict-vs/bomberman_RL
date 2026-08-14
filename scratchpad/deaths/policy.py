#!/usr/bin/env python3
"""At the last moment the death could still be dodged, what did the agent see?

`t*` = the last step of the window from which some own action sequence survives
(the opponents replay their recorded actions).  `A*` = the set of *first* actions
that begin such a sequence.  At `t*` the death is still avoidable by definition,
so everything that happens after it is the policy's doing -- and the question
becomes whether the eight digits and the shipped table could have produced an
action in `A*`.

Three outcomes are possible and they need different fixes:
  * the row is degenerate (every action has the same Q, usually an all-zero row):
    `act` falls through to a uniform random choice -- the *table* never learned
    this state, whatever the features say;
  * the row is trained and its greedy set lies inside `A*`: the agent should have
    survived and the loss is a tie-break;
  * the row is trained and its greedy set is fatal: the digits map this state
    onto a row whose learned answer kills here, i.e. the row is ambiguous.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import candidates  # noqa: E402
import sim  # noqa: E402

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
DIRS = {0: None, 1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
ROOT = Path(__file__).resolve().parents[2]
SIZES = (4, 4, 4, 4, 5, 5, 2, 5)


def decode(row: int) -> tuple:
    out = []
    for size in reversed(SIZES):
        out.append(row % size)
        row //= size
    return tuple(reversed(out))


def last_savable(win, dstep, max_k=5):
    """(t*, first actions that begin a surviving sequence). (None, set()) if none."""
    steps = [s["step"] for s in win]
    for k in range(1, max_k + 1):
        start = dstep - k + 1
        if start not in steps:
            return None, set()
        seq = [s for s in win if s["step"] >= start]
        base = sim.state_from_snapshot(seq[0])
        good = set()
        for combo in itertools.product(ACTIONS, repeat=k):
            if combo[0] in good:
                continue
            st = base.clone()
            for i, snap in enumerate(seq[:k]):
                st.do_step(sim.actions_from_snapshot(snap, combo[i]),
                           sim.order_from_snapshot(snap))
                if not st.alive("ME"):
                    break
            else:
                if sim.survivable(st):
                    good.add(combo[0])
        if good:
            return start, good
    return None, set()


def main() -> None:
    data = pickle.load(open(Path(__file__).parent / "deaths300.pkl", "rb"))
    q = np.load(ROOT / "agent_code" / "benedict_task4" / "q_table.npy")
    visits = Counter(e["row"] for e in data["step_log"])

    cats = Counter()
    p_surv, played_in_greedy, rows_at_t = [], 0, []
    degenerate_rows = Counter()
    cand_at_t = []
    for d in data["deaths"]:
        win = d["window"]
        t, good = last_savable(win, d["death_step"])
        if t is None:
            cats["not avoidable within 5 steps"] += 1
            continue
        snap = next(s for s in win if s["step"] == t)
        row = snap["row"]
        qrow = q[row]
        greedy = {ACTIONS[i] for i in np.flatnonzero(qrow >= qrow.max())}
        degen = bool(qrow.max() == qrow.min())
        played_in_greedy += int(snap["my_action"] in greedy)
        p = len(greedy & good) / len(greedy)
        p_surv.append(p)
        rows_at_t.append((row, t, d["death_step"], good, greedy, degen, snap, d["round"]))
        cand_at_t.append(snap["cand"])

        if degen:
            cats["row untrained -- every action ties, act() picks at random"] += 1
            degenerate_rows[row] += 1
        elif greedy <= good:
            cats["row trained, greedy action survives (lost to a tie-break)"] += 1
        elif greedy & good:
            cats["row trained, greedy set mixes survivors and killers"] += 1
        else:
            cats["row trained, every greedy action is fatal here"] += 1

    n = sum(cats.values())
    print(f"# deaths: {n}\n## what the shipped policy could do at t*, the last savable step")
    for k, v in cats.most_common():
        print(f"  {v:4d} ({v/n:5.1%})  {k}")
    print(f"\n  mean P(survive | shipped policy at t*) = {np.mean(p_surv):.3f}")
    print(f"  the action it actually played at t* was in the greedy set: "
          f"{played_in_greedy}/{len(p_surv)}  (sanity check on the feature recomputation)")

    print("\n## how far ahead t* was")
    print("  ", dict(sorted(Counter(t[2] - t[1] for t in rows_at_t).items())))

    print("\n## the untrained rows that kill most often")
    for row, cnt in degenerate_rows.most_common(8):
        print(f"    row {row:6d} digits={decode(row)}  deaths={cnt}  visits={visits[row]}")

    # -- what would a candidate digit have added AT t*? ----------------------
    print("\n## candidate feature values at t* vs. their base rate over all steps")
    log = data["step_log"]
    for i, name in enumerate(candidates.NAMES):
        at_t = Counter(c[i] for c in cand_at_t)
        base = Counter(e["cand"][i] for e in log)
        tot_b = sum(base.values())
        parts = "  ".join(f"{v}: {at_t[v]/len(cand_at_t):.2f} vs {base[v]/tot_b:.2f}"
                          for v in sorted(set(at_t) | set(base)))
        print(f"  {name:16s} {parts}")

    with open(Path(__file__).parent / "savable.pkl", "wb") as f:
        pickle.dump([{"round": rd, "row": r, "t": t, "d": dd, "good": sorted(g),
                      "greedy": sorted(gr), "degen": de}
                     for r, t, dd, g, gr, de, _, rd in rows_at_t], f)


if __name__ == "__main__":
    main()

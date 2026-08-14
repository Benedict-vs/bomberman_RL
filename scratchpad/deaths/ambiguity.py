#!/usr/bin/env python3
"""Split the two failure modes and price them.

(a) untrained rows: how often does the greedy policy fall through to a coin
    flip, and how much of that is specific to the death states?
(b) ambiguous rows: for the deaths where the row *is* trained and its answer is
    fatal, do the candidate digits separate the fatal visits of that row from
    the safe ones?  That is the "could the eight digits tell this apart"
    question, asked per row rather than in aggregate.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import candidates  # noqa: E402
import policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    data = pickle.load(open(Path(__file__).parent / "deaths300.pkl", "rb"))
    sav = pickle.load(open(Path(__file__).parent / "savable.pkl", "rb"))
    q = np.load(ROOT / "agent_code" / "benedict_task4" / "q_table.npy")

    degen_row = (q.max(axis=1) == q.min(axis=1))
    zero_row = (q == 0).all(axis=1)
    print(f"# table: {q.shape[0]} rows, {int((~degen_row).sum())} carry a decision, "
          f"{int((~zero_row).sum())} are non-zero")

    log = data["step_log"]
    d_steps = [e for e in log if degen_row[e["row"]]]
    print(f"# steps in a degenerate row: {len(d_steps)}/{len(log)} "
          f"({len(d_steps)/len(log):.3%}) -- the policy is a coin flip there")

    t_rows = [s["row"] for s in sav]
    print(f"# at t* (last savable step of a death): "
          f"{sum(degen_row[r] for r in t_rows)}/{len(t_rows)} "
          f"({np.mean([degen_row[r] for r in t_rows]):.1%})")

    # -- how fatal is a degenerate row when it is reached at all? ------------
    death_rows = Counter(s["row"] for s in sav)
    visits = Counter(e["row"] for e in log)
    dg = [r for r in death_rows if degen_row[r]]
    print(f"\n## the {len(dg)} degenerate rows that hold {sum(death_rows[r] for r in dg)} "
          f"of the {len(sav)} deaths")
    print(f"  they were visited {sum(visits[r] for r in dg)} times in total, so "
          f"{sum(death_rows[r] for r in dg)/max(sum(visits[r] for r in dg),1):.1%} of "
          f"every visit to one of them is the last savable step of a death")

    # -- (b) ambiguous rows ---------------------------------------------------
    amb = [s for s in sav if not s["degen"] and not (set(s["greedy"]) & set(s["good"]))]
    print(f"\n## {len(amb)} deaths where the row is trained and every greedy action is fatal")
    by_row = Counter(s["row"] for s in amb)
    print("  rows:", dict(by_row.most_common(10)))

    # For each such row, compare candidate values on the fatal visits with the
    # rest of the visits to the same row.
    fatal_at = defaultdict(set)      # row -> {(round, step)}
    death_of = {r["round"]: r["death_step"] for r in data["rounds"]}
    sav_keyed = {(s["row"], s["t"]) for s in amb}
    per_row_visits = defaultdict(list)
    for e in log:
        per_row_visits[e["row"]].append(e)

    print("\n  candidate separation inside those rows "
          "(value distribution on the fatal visit vs. the row's other visits)")
    for row, cnt in by_row.most_common(6):
        es = per_row_visits[row]
        fatal = [e for e in es
                 if death_of.get(e["round"]) is not None
                 and 0 <= death_of[e["round"]] - e["step"] <= 4]
        rest = [e for e in es if e not in fatal]
        line = []
        for i, name in enumerate(candidates.NAMES):
            fv = Counter(e["cand"][i] for e in fatal)
            rv = Counter(e["cand"][i] for e in rest)
            if not fv or not rv:
                continue
            # only report a candidate whose modal fatal value is rare elsewhere
            v, c = fv.most_common(1)[0]
            p_f = c / len(fatal)
            p_r = rv[v] / len(rest)
            if p_f > 0.6 and p_r < 0.3:
                line.append(f"{name}={v} ({p_f:.0%} fatal vs {p_r:.0%} safe)")
        print(f"    row {row:6d} deaths={cnt} visits={len(es)} "
              f"fatal_steps={len(fatal)}  digits={policy.decode(row)}")
        print(f"      separators: {', '.join(line) if line else 'none of the candidates'}")


if __name__ == "__main__":
    main()

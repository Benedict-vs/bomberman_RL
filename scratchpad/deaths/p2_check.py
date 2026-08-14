"""E30 P2: does training in the rung-4 field empty the all-zero rows we die in?

E28's forensic measured, for the *frozen* table, that 73.4 % of deaths reach the
last savable step `t*` in a row where every action ties -- an all-zero row, where
`act()` draws uniformly over six actions. E30 pre-registered that training in the
`rule_based` field drives that share **below 40 %**, with refutation above 60 %.

This re-runs exactly that classification against a different table. It reuses
`policy.last_savable` rather than reimplementing it, so the definition of `t*` is
the same object E28 measured and the two numbers are comparable.

    uv run python scratchpad/deaths/p2_check.py --deaths <pkl> --table <npy>

The deaths pickle must be collected with the *same* table: `t*` depends on the
policy, so classifying one table's deaths against another's Q-values would be
measuring nothing.
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

import policy                       # same directory; supplies last_savable + ACTIONS


def classify(deaths_pkl: Path, table: Path) -> dict:
    data = pickle.load(open(deaths_pkl, "rb"))
    q = np.load(table)

    cats = Counter()
    for d in data["deaths"]:
        win = d["window"]
        t, good = policy.last_savable(win, d["death_step"])
        if t is None:
            cats["not avoidable within 5 steps"] += 1
            continue
        snap = next(s for s in win if s["step"] == t)
        qrow = q[snap["row"]]
        greedy = {policy.ACTIONS[i] for i in np.flatnonzero(qrow >= qrow.max())}

        if qrow.max() == qrow.min():
            cats["degenerate row -- uniform draw"] += 1
        elif greedy <= good:
            cats["trained, greedy survives (lost to a tie-break)"] += 1
        elif greedy & good:
            cats["trained, greedy set mixes survivors and killers"] += 1
        else:
            cats["trained, every greedy action is fatal"] += 1

    n = sum(cats.values())
    return {"n": n, "cats": cats,
            "degenerate_share": cats["degenerate row -- uniform draw"] / n if n else float("nan"),
            "nonzero_rows": int((np.abs(q).sum(axis=1) > 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deaths", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    r = classify(args.deaths, args.table)
    print(f"{args.label or args.table.name}   deaths={r['n']}  "
          f"nonzero rows={r['nonzero_rows']}")
    for k, v in r["cats"].most_common():
        print(f"    {v:4d}  {100*v/r['n']:5.1f} %  {k}")
    print(f"  --> degenerate share at t* = {100*r['degenerate_share']:.1f} % "
          f"(E28 frozen: 73.4 %; P2 passes below 40 %, refuted above 60 %)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What would an extra feature digit have bought?

Two questions, both answered on the *same* 300-round run the deaths come from:

1. Raw signal -- P(dead within k steps | candidate value) against the base rate.
   A candidate with no lift cannot help whatever the learner is.
2. Signal the agent does not already have -- the same probability *within* a
   fixed row of the current 8-digit map. If a row's visits split into a fatal
   and a safe group along the candidate, the current features genuinely cannot
   tell those situations apart and the table is being asked an impossible
   question. If they do not split, the row is fine and the loss is the policy's.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import candidates  # noqa: E402

HORIZON = 4          # "about to die": within one bomb timer


def label(data) -> list[dict]:
    """Attach `to_death` (steps until our agent was removed) to every step."""
    death_of = {r["round"]: r["death_step"] for r in data["rounds"]}
    out = []
    for e in data["step_log"]:
        d = death_of.get(e["round"])
        e = dict(e)
        e["to_death"] = None if d is None else d - e["step"]
        e["fatal"] = int(e["to_death"] is not None and 0 <= e["to_death"] <= HORIZON)
        out.append(e)
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "deaths300.pkl"
    data = pickle.load(open(path, "rb"))
    log = label(data)
    n = len(log)
    base = np.mean([e["fatal"] for e in log])
    print(f"# steps={n}  P(dead within {HORIZON} steps) = {base:.4f}\n")

    print("## 1. raw signal per candidate")
    for i, name in enumerate(candidates.NAMES):
        groups = defaultdict(list)
        for e in log:
            groups[e["cand"][i]].append(e["fatal"])
        parts = "  ".join(
            f"{v}: {np.mean(g):.3f} (n={len(g)})" for v, g in sorted(groups.items()))
        print(f"  {name:16s} {parts}")

    print("\n## 2. within-row lift  (rows with >=200 visits, the candidate splits them)")
    by_row = defaultdict(list)
    for e in log:
        by_row[e["row"]].append(e)

    for i, name in enumerate(candidates.NAMES):
        # Deaths that land in a row where the candidate separates fatal from safe
        gained = 0          # fatal steps whose (row, cand) cell is >=3x the row rate
        tot_fatal = 0
        best = []
        for row, es in by_row.items():
            if len(es) < 200:
                continue
            row_rate = np.mean([e["fatal"] for e in es])
            if row_rate == 0:
                continue
            cells = defaultdict(list)
            for e in es:
                cells[e["cand"][i]].append(e["fatal"])
            if len(cells) < 2:
                continue
            for v, g in cells.items():
                r = np.mean(g)
                f = sum(g)
                tot_fatal += f
                if r >= 3 * row_rate:
                    gained += f
                    best.append((f, row, name, v, r, row_rate, len(g), len(es)))
        best.sort(reverse=True)
        print(f"  {name:16s} fatal steps moved into a >=3x cell: {gained:5d} / {tot_fatal:5d}"
              f"  ({gained / max(tot_fatal, 1):5.1%})")
        for f, row, nm, v, r, rr, ng, ne in best[:3]:
            print(f"      row {row:6d} (n={ne:5d}, rate {rr:.3f}) -> {nm}={v}: "
                  f"rate {r:.3f} on n={ng} ({f} fatal steps)")

    # -- the pairwise version: which single digit would have caught most deaths --
    print("\n## 3. how many of the death steps themselves the candidate flags")
    deaths = [e for e in log if e["to_death"] == 0]
    near = [e for e in log if e["to_death"] is not None and 1 <= e["to_death"] <= 3]
    safe = [e for e in log if e["to_death"] is None or e["to_death"] > 8]
    for i, name in enumerate(candidates.NAMES):
        dv = Counter(e["cand"][i] for e in deaths)
        nv = Counter(e["cand"][i] for e in near)
        sv = Counter(e["cand"][i] for e in safe)
        print(f"  {name:16s} at death {dict(sorted(dv.items()))}")
        print(f"  {'':16s} 1-3 before {dict(sorted(nv.items()))}")
        print(f"  {'':16s} safe steps {dict(sorted(sv.items()))}")


if __name__ == "__main__":
    main()

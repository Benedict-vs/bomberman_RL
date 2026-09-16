#!/usr/bin/env python3
"""Detect a plateau from chronological greedy evaluation CSVs.

Training score/loss is deliberately not used. Pass evaluation CSVs in
chronological order; each must contain per-round rows for the selected agent.
Exit status: 0 = plateau, 1 = still improving/insufficient history, 2 = error.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_mean(path: Path, agent: str) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("agent") == agent]
    if not rows:
        raise ValueError(f"no rows for agent {agent!r} in {path}")

    def mean(name: str) -> float:
        values = [float(row[name]) for row in rows]
        return sum(values) / len(values)

    return {
        "score": mean("score"),
        "kills": mean("kills"),
        "suicides": mean("suicides"),
        "survived": mean("survived"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--agent", default="ben_task4")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-score-delta", type=float, default=0.10)
    parser.add_argument("--max-suicide-regression", type=float, default=0.03)
    args = parser.parse_args(argv)

    if args.patience < 1 or len(args.csv) < args.patience + 1:
        print(
            f"PLATEAU=insufficient_history files={len(args.csv)} "
            f"needed={args.patience + 1}"
        )
        return 1

    try:
        summaries = [read_mean(path, args.agent) for path in args.csv]
    except (OSError, ValueError, KeyError) as exc:
        print(f"plateau_check error: {exc}", file=sys.stderr)
        return 2

    print("phase,score,kills,suicides,survived")
    for index, summary in enumerate(summaries, start=1):
        print(
            f"{index},{summary['score']:.3f},{summary['kills']:.3f},"
            f"{summary['suicides']:.3f},{summary['survived']:.3f}"
        )

    recent = summaries[-(args.patience + 1) :]
    score_deltas = [
        recent[index + 1]["score"] - recent[index]["score"]
        for index in range(args.patience)
    ]
    suicide_deltas = [
        recent[index + 1]["suicides"] - recent[index]["suicides"]
        for index in range(args.patience)
    ]
    plateau = all(delta < args.min_score_delta for delta in score_deltas)
    safety_regression = any(
        delta > args.max_suicide_regression for delta in suicide_deltas
    )
    status = "plateau" if plateau else "still_improving"
    print(
        f"PLATEAU={status} score_deltas="
        f"{','.join(f'{delta:+.3f}' for delta in score_deltas)} "
        f"suicide_deltas={','.join(f'{delta:+.3f}' for delta in suicide_deltas)} "
        f"safety_regression={str(safety_regression).lower()}"
    )
    return 0 if plateau else 1


if __name__ == "__main__":
    raise SystemExit(main())


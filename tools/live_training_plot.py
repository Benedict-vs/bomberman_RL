#!/usr/bin/env python3
"""Refresh a training-progress diagram after each completed episode block."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float]]:
    """Read complete numeric rows while tolerating a concurrently written CSV."""
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({key: float(value) for key, value in row.items()})
            except (TypeError, ValueError):
                continue
    return rows


def block_means(rows, column: str, block_size: int):
    """Return episode endpoints and non-overlapping block means."""
    endpoints, means = [], []
    for start in range(0, len(rows), block_size):
        block = rows[start:start + block_size]
        if not block:
            continue
        values = [row[column] for row in block]
        endpoints.append(int(block[-1]["episode"]))
        means.append(sum(values) / len(values))
    return endpoints, means


def draw(rows, output: Path, block_size: int) -> None:
    """Write the current dashboard atomically so viewers never see half a PNG."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        (axes[0, 0], [("COIN_COLLECTED", "Münzen"),
                      ("CRATE_DESTROYED", "Kisten")], "Ereignisse / Episode"),
        (axes[0, 1], [("KILLED_SELF", "Suizid"),
                      ("SURVIVED_ROUND", "Überlebt")], "Anteil"),
        (axes[1, 0], [("steps", "Schritte")], "Schritte / Episode"),
        (axes[1, 1], [("BOMB_DROPPED", "Bomben"),
                      ("WAITED", "WAIT")], "Aktionen / Episode"),
        (axes[2, 0], [("epsilon", "Epsilon")], "Epsilon"),
        (axes[2, 1], [("mean_loss", "Loss")], "Mittlerer Loss"),
    ]
    for axis, series, ylabel in panels:
        for column, label in series:
            x, y = block_means(rows, column, block_size)
            axis.plot(x, y, marker="o", markersize=2.5, linewidth=1.4,
                      label=label)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if len(series) > 1:
            axis.legend()
    fig.suptitle(
        f"Task-2-Training – Mittelwerte je {block_size} Episoden "
        f"(Stand: Episode {int(rows[-1]['episode'])})"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.png")
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--stop-episode", type=int)
    args = parser.parse_args()

    last_completed_block = -1
    while True:
        rows = read_rows(args.csv)
        if rows:
            completed_block = int(rows[-1]["episode"]) // args.block_size
            if completed_block != last_completed_block:
                draw(rows, args.output, args.block_size)
                print(f"Diagramm aktualisiert: Episode {int(rows[-1]['episode'])}",
                      flush=True)
                last_completed_block = completed_block
            if (args.stop_episode is not None
                    and rows[-1]["episode"] >= args.stop_episode):
                return
        if not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

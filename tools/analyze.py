#!/usr/bin/env python3
"""Turn evaluation CSVs into numbers we can defend in the report.

Two modes:

**Summary** -- mean and 95 % bootstrap confidence interval per agent::

    uv run python tools/analyze.py results/eval/q_v3.csv

**Comparison** -- paired difference between two runs::

    uv run python tools/analyze.py --compare results/eval/q_v2.csv \\
        results/eval/q_v3.csv --agent schmaxi_coin_collector

Add ``--markdown`` to get a table that pastes straight into the report.

Why bootstrap, why paired
-------------------------
Per-round score in Bomberman is wildly skewed: many rounds near zero, a few
large ones when kills happen. The normal approximation for a confidence interval
assumes something this distribution does not satisfy, so we resample instead --
no distributional assumption, and it behaves well on the tail.

Paired comparison exploits the fact that ``evaluate.py`` gives both runs the
*same* arenas (see the arena-matching note there). Instead of comparing two
noisy means we take the difference round by round, which cancels "this arena was
generous" and leaves the effect of the change. The CI on the paired difference
is usually several times narrower than on either mean -- which is often the
difference between "we measured an improvement" and "we cannot tell".

What counts as a result
-----------------------
If the 95 % CI of the paired difference contains 0, the change is *not*
demonstrated. Say so in the report -- a well-documented negative result is worth
more marks than an unsupported claim of improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# metric -> (column, label, higher_is_better, format)
METRICS = {
    "score":    ("score",          "Score",          True,  "{:.3f}"),
    "coins":    ("coins",          "Coins",          True,  "{:.3f}"),
    "kills":    ("kills",          "Kills",          True,  "{:.3f}"),
    "suicides": ("suicides",       "Suicides",       False, "{:.3f}"),
    "crates":   ("crates",         "Crates",         True,  "{:.2f}"),
    "bombs":    ("bombs",          "Bombs",          True,  "{:.2f}"),
    "survived": ("survived",       "Survival rate",  True,  "{:.3f}"),
    "steps":    ("steps",          "Steps alive",    True,  "{:.1f}"),
    "invalid":  ("invalid",        "Invalid actions", False, "{:.2f}"),
    "think_ms": ("think_max_ms",   "Think max (ms)", False, "{:.1f}"),
}

DEFAULT_METRICS = ["score", "coins", "kills", "suicides", "survived", "invalid"]

INT_COLUMNS = {"round", "seed", "slot", "survived", "round_steps",
               "score", "coins", "kills", "suicides", "crates", "bombs",
               "moves", "invalid", "steps", "think_over_limit"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No such file: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"Empty file: {path}")
    for row in rows:
        for key, value in row.items():
            if key in INT_COLUMNS:
                row[key] = int(value)
            elif key not in ("agent", "code"):
                row[key] = float(value)
    return rows


def load_meta(csv_path: Path) -> dict:
    meta_path = csv_path.with_suffix("").with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = csv_path.parent / f"{csv_path.stem}.meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    return {}


def rows_for(rows: list[dict], agent: str | None) -> list[dict]:
    """Rows for one agent. Without a name, take the first slot (ours by
    convention -- evaluate.py puts our agent first)."""
    if agent is None:
        names = [r["agent"] for r in rows if r["slot"] == 0]
        if not names:
            raise SystemExit("No slot-0 agent found.")
        agent = names[0]
    selected = [r for r in rows if r["agent"] == agent]
    if not selected:
        available = sorted({r["agent"] for r in rows})
        raise SystemExit(f"Agent {agent!r} not in file. Available: {available}")
    return selected


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, low, high)."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    rng = rng or np.random.default_rng(12345)  # fixed: same data -> same CI
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(low), float(high)


def paired_series(rows_a: list[dict], rows_b: list[dict], column: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Values from both runs restricted to the seeds they share."""
    map_a = {r["seed"]: r[column] for r in rows_a}
    map_b = {r["seed"]: r[column] for r in rows_b}
    shared = sorted(set(map_a) & set(map_b))
    if not shared:
        raise SystemExit(
            "The two runs share no seeds, so they cannot be paired. Re-run both "
            "with the same --seed and --n-rounds."
        )
    return (np.array([map_a[s] for s in shared], dtype=float),
            np.array([map_b[s] for s in shared], dtype=float))


def wilcoxon_p(differences: np.ndarray) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value, if scipy is installed."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    nonzero = differences[differences != 0]
    if len(nonzero) < 10:
        return None
    try:
        return float(wilcoxon(nonzero).pvalue)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarise(path: Path, metrics: list[str], markdown: bool, n_boot: int) -> None:
    rows = load(path)
    meta = load_meta(path)
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent"]].append(row)

    n_rounds = len({r["round"] for r in rows})
    header = f"{path.name} -- {n_rounds} rounds"
    if meta:
        header += (f", scenario={meta.get('scenario')}, "
                   f"commit={meta.get('git_commit')}")

    if markdown:
        print(f"\n**{header}**\n")
        print("| Agent | " + " | ".join(METRICS[m][1] for m in metrics) + " |")
        print("|" + "---|" * (len(metrics) + 1))
        for agent, agent_rows in by_agent.items():
            cells = []
            for metric in metrics:
                column, _, _, fmt = METRICS[metric]
                mean, low, high = bootstrap_ci(
                    np.array([r[column] for r in agent_rows]), n_boot)
                cells.append(f"{fmt.format(mean)} [{fmt.format(low)}, {fmt.format(high)}]")
            print(f"| `{agent}` | " + " | ".join(cells) + " |")
        print("\n<sub>Mean with 95 % bootstrap CI over rounds.</sub>")
    else:
        print(f"\n{header}")
        print("=" * len(header))
        for agent, agent_rows in by_agent.items():
            print(f"\n  {agent}  (n={len(agent_rows)})")
            for metric in metrics:
                column, label, _, fmt = METRICS[metric]
                mean, low, high = bootstrap_ci(
                    np.array([r[column] for r in agent_rows]), n_boot)
                print(f"    {label:<16} {fmt.format(mean):>9}   "
                      f"95% CI [{fmt.format(low)}, {fmt.format(high)}]")
        over = sum(r.get("think_over_limit", 0) for r in rows)
        if over:
            print(f"\n  WARNING: {over} steps exceeded the time limit.")
        print()


def compare(path_a: Path, path_b: Path, agent: str | None, metrics: list[str],
            markdown: bool, n_boot: int) -> None:
    rows_a = rows_for(load(path_a), agent)
    rows_b = rows_for(load(path_b), agent)
    name_a, name_b = path_a.stem, path_b.stem

    lines = []
    for metric in metrics:
        column, label, higher_better, fmt = METRICS[metric]
        values_a, values_b = paired_series(rows_a, rows_b, column)
        differences = values_b - values_a

        mean_diff, low, high = bootstrap_ci(differences, n_boot)
        significant = (low > 0) or (high < 0)
        improved = (mean_diff > 0) == higher_better
        if not significant:
            verdict = "no effect shown"
        elif improved:
            verdict = "BETTER"
        else:
            verdict = "WORSE"

        p_value = wilcoxon_p(differences)
        win_rate = float(np.mean(values_b > values_a))

        lines.append({
            "label": label, "fmt": fmt,
            "a": values_a.mean(), "b": values_b.mean(),
            "diff": mean_diff, "low": low, "high": high,
            "verdict": verdict, "p": p_value, "win": win_rate,
            "n": len(differences),
        })

    n_paired = lines[0]["n"]
    agent_a, agent_b = rows_a[0]["agent"], rows_b[0]["agent"]
    who = agent_a if agent_a == agent_b else f"{agent_a} -> {agent_b}"
    title = f"{name_b}  vs  {name_a}   (agent: {who}, {n_paired} paired rounds)"

    if markdown:
        print(f"\n**{title}**\n")
        print("| Metric | " + f"{name_a} | {name_b} | Difference (95 % CI) | Verdict |")
        print("|---|---|---|---|---|")
        for line in lines:
            fmt = line["fmt"]
            print(f"| {line['label']} | {fmt.format(line['a'])} | "
                  f"{fmt.format(line['b'])} | "
                  f"{line['diff']:+.3f} [{line['low']:+.3f}, {line['high']:+.3f}] | "
                  f"{line['verdict']} |")
        print("\n<sub>Paired difference (B − A) on identical arenas, "
              "95 % bootstrap CI. A CI containing 0 means the change is not "
              "demonstrated at this sample size.</sub>")
    else:
        print(f"\n{title}")
        print("=" * len(title))
        print(f"\n  {'Metric':<16}{'A':>9}{'B':>9}{'diff':>10}"
              f"{'95% CI':>22}   verdict")
        print("  " + "-" * 76)
        for line in lines:
            fmt = line["fmt"]
            ci = f"[{line['low']:+.3f}, {line['high']:+.3f}]"
            print(f"  {line['label']:<16}{fmt.format(line['a']):>9}"
                  f"{fmt.format(line['b']):>9}{line['diff']:>+10.3f}{ci:>22}   "
                  f"{line['verdict']}")
            extra = f"    B better in {line['win']:.0%} of rounds"
            if line["p"] is not None:
                extra += f", Wilcoxon p={line['p']:.4g}"
            print(extra)
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise or compare evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", type=Path,
                        help="Evaluation CSVs from tools/evaluate.py")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "NEW"),
                        help="Paired comparison: BASE against NEW.")
    parser.add_argument("--agent", default=None,
                        help="Which agent to compare. Default: slot 0.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        choices=sorted(METRICS),
                        help=f"Default: {' '.join(DEFAULT_METRICS)}")
    parser.add_argument("--markdown", action="store_true",
                        help="Emit a Markdown table for the report.")
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Bootstrap resamples (default 10000).")

    args = parser.parse_args(argv)

    if not args.files and not args.compare:
        parser.error("Give at least one CSV, or use --compare BASE NEW.")

    if args.compare:
        compare(args.compare[0], args.compare[1], args.agent,
                args.metrics, args.markdown, args.n_boot)
    for path in args.files:
        summarise(path, args.metrics, args.markdown, args.n_boot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

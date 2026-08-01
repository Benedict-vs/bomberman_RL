#!/usr/bin/env python3
"""

Two options: summarize a single run, or compare two runs

**Summary** -- mean and 95 % bootstrap confidence interval per agent::

    uv run python tools/analyze.py results/eval/q_v3.csv

**Comparison** -- paired difference between two runs:

    uv run python tools/analyze.py --compare results/eval/q_v2.csv \\
        results/eval/q_v3.csv --agent (agent name)

**Ablation**  one metric, many variants, all against the same baseline, see if one component is responsible for the effect:

    uv run python tools/analyze.py --ablation results/eval/q_base.csv \\
        results/eval/q_no_shaping.csv results/eval/q_no_symmetry.csv \\
        --metric (add metric name/s ) --plot

Add ``--markdown`` for a table that pastes straight into the report, ``--plot``
for a figure (bar chart with CIs for a summary, forest plot for a comparison or
ablation). Figures land in ``results/figures/``.

bootsrap and paired comparison are used because the per-round score distribution is not normal.

Per-round score in Bomberman is wildly skewed: many rounds near zero, a few
large ones when kills happen. The normal approximation for a confidence interval
assumes something this distribution does not satisfy, so we resample instead 
no distributional assumption, and it behaves well on the tail.

Paired comparison uses the fact that ``evaluate.py`` gives both runs the
same arenas #. Instead of comparing two
noisy means we take the difference round by round, which cancels random arena differences 
and leaves the effect of the change.


What counts as a result:

If the 95 % CI of the paired difference contains 0, the change is not
demonstrated. Does not mean it is not better, just that we cannot tell with this sample size.
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
    "score":     ("score",             "Score",            True,  "{:.3f}"),
    "won":       ("won",               "Win rate",         True,  "{:.3f}"),
    "rank":      ("rank",              "Rank (1 = best)",  False, "{:.2f}"),
    "coins":     ("coins",             "Coins",            True,  "{:.3f}"),
    "kills":     ("kills",             "Kills",            True,  "{:.3f}"),
    "suicides":  ("suicides",          "Suicides",         False, "{:.3f}"),
    "killed_by": ("killed_by_opponent", "Killed by opp.",  False, "{:.3f}"),
    "died":      ("died",              "Death rate",       False, "{:.3f}"),
    "crates":    ("crates",            "Crates",           True,  "{:.2f}"),
    "bombs":     ("bombs",             "Bombs",            True,  "{:.2f}"),
    "survived":  ("survived",          "Survival rate",    True,  "{:.3f}"),
    "steps":     ("steps",             "Steps alive",      True,  "{:.1f}"),
    "invalid":   ("invalid",           "Invalid actions",  False, "{:.2f}"),
    "think_ms":  ("think_max_ms",      "Think max (ms)",   False, "{:.1f}"),
}

DEFAULT_METRICS = ["score", "coins", "kills", "suicides", "survived", "invalid"]

# presets for the task ladder each stage keeps the
# diagnostics of the one before it; `suicides` stops being the progress signal
# after stage 2 but stays in as a regression guard, because learning to hunt is
# exactly when an agent starts forgetting to run from its own bomb.
PRESETS = {
    "task1": ["coins", "steps", "invalid"],
    "task2": ["score", "suicides", "crates", "bombs", "survived"],
    "task3": ["score", "kills", "suicides", "survived", "invalid"],
    "task4": ["score", "won", "kills", "suicides", "killed_by", "think_ms"],
}

INT_COLUMNS = {"round", "seed", "slot", "survived", "round_steps",
               "score", "coins", "kills", "suicides", "crates", "bombs",
               "moves", "invalid", "steps", "think_over_limit",
               "died", "killed_by_opponent", "rank", "won"}


#loading
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
    _backfill(rows)
    return rows


def _backfill(rows: list[dict]) -> None:
    """Derive the newer columns for CSVs written before they existed.

    Keeps older result files usable instead of forcing a re-run of every
    measurement whenever we add a metric.
    """
    if "died" not in rows[0] and "survived" in rows[0]:
        for row in rows:
            row["died"] = 1 - row["survived"]
    if "killed_by_opponent" not in rows[0] and "died" in rows[0]:
        for row in rows:
            row["killed_by_opponent"] = max(0, row["died"] - row.get("suicides", 0))
    if "rank" not in rows[0] and "score" in rows[0]:
        by_round: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_round[row["round"]].append(row)
        for group in by_round.values():
            best = max(r["score"] for r in group)
            for row in group:
                row["rank"] = 1 + sum(1 for o in group if o["score"] > row["score"])
                row["won"] = int(row["score"] == best)


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
    """Two-sided Wilcoxon signed-rank p-value"""
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



# Reporting

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


def paired_effect(rows_a: list[dict], rows_b: list[dict], metric: str,
                  n_boot: int) -> dict:
    """Paired difference (B − A) for one metric, with verdict."""
    column, label, higher_better, fmt = METRICS[metric]
    values_a, values_b = paired_series(rows_a, rows_b, column)
    differences = values_b - values_a

    mean_diff, low, high = bootstrap_ci(differences, n_boot)
    significant = (low > 0) or (high < 0)
    improved = (mean_diff > 0) == higher_better
    verdict = "no effect shown" if not significant else ("BETTER" if improved else "WORSE")

    return {
        "metric": metric, "label": label, "fmt": fmt,
        "a": float(values_a.mean()), "b": float(values_b.mean()),
        "diff": mean_diff, "low": low, "high": high,
        "verdict": verdict,
        "p": wilcoxon_p(differences),
        "win": float(np.mean(values_b > values_a)),
        "n": len(differences),
    }


def compare(path_a: Path, path_b: Path, agent: str | None, metrics: list[str],
            markdown: bool, n_boot: int, plot: Path | None = None) -> None:
    rows_a = rows_for(load(path_a), agent)
    rows_b = rows_for(load(path_b), agent)
    name_a, name_b = path_a.stem, path_b.stem

    lines = [paired_effect(rows_a, rows_b, metric, n_boot) for metric in metrics]

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

    if plot is not None:
        plot_forest(
            entries=[{"name": line["label"], **line} for line in lines],
            title=f"{name_b} vs {name_a}  ({n_paired} paired rounds)",
            xlabel=f"paired difference  ({name_b} − {name_a})",
            out=plot,
        )


def ablation(base: Path, variants: list[Path], metric: str, agent: str | None,
             n_boot: int, markdown: bool, plot: Path | None) -> None:
    """one metric, many variants, all measured against the same baseline.
    """
    rows_base = rows_for(load(base), agent)
    entries = []
    for path in variants:
        effect = paired_effect(rows_base, rows_for(load(path), agent), metric, n_boot)
        effect["name"] = path.stem
        entries.append(effect)

    label = METRICS[metric][1]
    title = f"{label}: variants vs {base.stem}"

    if markdown:
        print(f"\n**{title}** ({entries[0]['n']} paired rounds)\n")
        print(f"| Variant | {label} | Difference vs baseline (95 % CI) | Verdict |")
        print("|---|---|---|---|")
        for entry in entries:
            fmt = entry["fmt"]
            print(f"| `{entry['name']}` | {fmt.format(entry['b'])} | "
                  f"{entry['diff']:+.3f} [{entry['low']:+.3f}, {entry['high']:+.3f}] | "
                  f"{entry['verdict']} |")
        print(f"\n<sub>Baseline `{base.stem}`: {entries[0]['fmt'].format(entries[0]['a'])}. "
              f"Paired on identical arenas, 95 % bootstrap CI.</sub>")
    else:
        print(f"\n{title}")
        print("=" * len(title))
        print(f"  baseline {base.stem}: {entries[0]['fmt'].format(entries[0]['a'])}"
              f"   ({entries[0]['n']} paired rounds)\n")
        width = max(len(e["name"]) for e in entries) + 2
        for entry in entries:
            ci = f"[{entry['low']:+.3f}, {entry['high']:+.3f}]"
            print(f"  {entry['name']:<{width}}{entry['diff']:>+9.3f}{ci:>22}   "
                  f"{entry['verdict']}")
        print()

    if plot is not None:
        plot_forest(entries, title, f"paired difference in {label}", plot)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
VERDICT_COLOURS = {
    "BETTER": "tab:green",
    "WORSE": "tab:red",
    "no effect shown": "tab:gray",
}


def _pyplot():
    """matplotlib is analysis-only -- install with `uv add matplotlib`."""
    try:
        import matplotlib
    except ImportError:
        raise SystemExit(
            "matplotlib is required for --plot.  Install it with:  uv add matplotlib"
        )
    matplotlib.use("Agg")           # no display needed, writes straight to file
    import matplotlib.pyplot as plt
    return plt


def plot_forest(entries: list[dict], title: str, xlabel: str, out: Path) -> Path:
    """Forest plot: one row per entry, point estimate with 95 % CI, zero line.
    """
    plt = _pyplot()

    entries = list(entries)[::-1]        # first entry ends up on top
    positions = np.arange(len(entries))
    diffs = np.array([e["diff"] for e in entries])
    lower = diffs - np.array([e["low"] for e in entries])
    upper = np.array([e["high"] for e in entries]) - diffs
    colours = [VERDICT_COLOURS[e["verdict"]] for e in entries]

    height = max(2.2, 0.55 * len(entries) + 1.4)
    fig, ax = plt.subplots(figsize=(7.5, height))

    ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=1)
    for pos, diff, low, high, colour in zip(positions, diffs, lower, upper, colours):
        ax.errorbar(diff, pos, xerr=[[low], [high]], fmt="o", color=colour,
                    capsize=4, markersize=7, linewidth=1.8, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels([e["name"] for e in entries])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.6, len(entries) - 0.4)

    handles = [plt.Line2D([], [], color=colour, marker="o", linestyle="",
                          label=verdict)
               for verdict, colour in VERDICT_COLOURS.items()]
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_summary(path: Path, metrics: list[str], agent: str | None,
                 n_boot: int, out: Path) -> Path:
    """Bar chart per metric, one bar per agent, error bars = 95 % bootstrap CI."""
    plt = _pyplot()

    rows = load(path)
    if agent:
        rows = [r for r in rows if r["agent"] == agent]
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent"]].append(row)
    names = list(by_agent)

    n = len(metrics)
    cols = min(3, n)
    plot_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(plot_rows, cols,
                             figsize=(4.2 * cols, 3.4 * plot_rows),
                             squeeze=False)

    for index, metric in enumerate(metrics):
        ax = axes[index // cols][index % cols]
        column, label, _, _ = METRICS[metric]
        means, lows, highs = [], [], []
        for name in names:
            mean, low, high = bootstrap_ci(
                np.array([r[column] for r in by_agent[name]]), n_boot)
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)

        ax.bar(range(len(names)), means, yerr=[lows, highs], capsize=4,
               color="tab:blue", alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    for index in range(n, plot_rows * cols):       #  blank out unused panels
        axes[index // cols][index % cols].axis("off")

    n_rounds = len({r["round"] for r in rows})
    fig.suptitle(f"{path.stem} — {n_rounds} rounds, mean with 95 % bootstrap CI",
                 fontsize=11)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise or compare evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", type=Path,
                        help="Evaluation CSVs from tools/evaluate.py")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "NEW"),
                        help="Paired comparison: BASE against NEW.")
    parser.add_argument("--ablation", nargs="+", type=Path,
                        metavar=("BASE", "VARIANT"),
                        help="One baseline plus any number of variants, all "
                             "compared on a single metric (see --metric).")
    parser.add_argument("--metric", default="score", choices=sorted(METRICS),
                        help="The single metric used by --ablation.")
    parser.add_argument("--agent", default=None,
                        help="Which agent to analyse. Default: slot 0.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        choices=sorted(METRICS),
                        help=f"Default: {' '.join(DEFAULT_METRICS)}")
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS),
                        help="Metric set for a rung of the task ladder: "
                             + "; ".join(f"{k}={' '.join(v)}" for k, v in PRESETS.items()))
    parser.add_argument("--markdown", action="store_true",
                        help="Emit a Markdown table for the report.")
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        metavar="PATH",
                        help="Also write a figure. Without a path it is placed "
                             "in results/figures/. Needs matplotlib "
                             "(uv add matplotlib).")
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Bootstrap resamples (default 10000).")

    args = parser.parse_args(argv)

    if args.metrics and args.preset:
        parser.error("Use either --metrics or --preset, not both.")
    args.metrics = args.metrics or (PRESETS[args.preset] if args.preset
                                    else DEFAULT_METRICS)

    if not args.files and not args.compare and not args.ablation:
        parser.error("Give at least one CSV, or use --compare / --ablation.")
    if args.ablation and len(args.ablation) < 2:
        parser.error("--ablation needs a baseline and at least one variant.")

    figure_dir = REPO_ROOT / "results" / "figures"

    def figure_path(stem: str) -> Path | None:
        if args.plot is None:
            return None
        return figure_dir / f"{stem}.png" if args.plot == "auto" else Path(args.plot)

    if args.compare:
        base, new = args.compare
        compare(base, new, args.agent, args.metrics, args.markdown, args.n_boot,
                plot=figure_path(f"compare_{base.stem}_vs_{new.stem}"))

    if args.ablation:
        base, *variants = args.ablation
        ablation(base, variants, args.metric, args.agent, args.n_boot,
                 args.markdown, plot=figure_path(f"ablation_{args.metric}"))

    for path in args.files:
        summarise(path, args.metrics, args.markdown, args.n_boot)
        target = figure_path(f"summary_{path.stem}")
        if target is not None:
            plot_summary(path, args.metrics, args.agent, args.n_boot, target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

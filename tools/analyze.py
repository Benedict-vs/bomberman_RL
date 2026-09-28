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
# LaTeX output
# --------------------------------------------------------------------------
# Emits complete booktabs tables to paste into the report. Deliberately uses
# nothing beyond `booktabs`, which the report preamble already loads -- no
# siunitx, no makecell, no threeparttable, so a copied table always compiles.
#
# Cells are written as  $3.060_{[2.700,\,3.400]}$: the estimate stays readable
# at a glance and the interval rides along without doubling the column count.

def tex_escape(text: str) -> str:
    """Escape the characters that actually occur in agent and file names."""
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def tex_code(text: str) -> str:
    return r"\texttt{" + tex_escape(text) + "}"


def tex_ci(fmt: str, mean: float, low: float, high: float, signed=False) -> str:
    """`$mean_{[low, high]}$`, optionally with explicit + signs."""
    if signed:
        body = f"{mean:+.3f}_{{[{low:+.3f},\\,{high:+.3f}]}}"
    else:
        body = (f"{fmt.format(mean)}_{{[{fmt.format(low)},\\,"
                f"{fmt.format(high)}]}}")
    return f"${body}$"


def tex_table(caption: str, label: str, column_spec: str,
              header: list[str], rows: list[list[str]]) -> None:
    print()
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(f"  \\caption{{{caption}}}")
    print(f"  \\label{{{label}}}")
    print(f"  \\begin{{tabular}}{{{column_spec}}}")
    print(r"    \toprule")
    print("    " + " & ".join(header) + r" \\")
    print(r"    \midrule")
    for row in rows:
        print("    " + " & ".join(row) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()


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


def signflip_p(differences: np.ndarray, n_perm: int = 20_000,
               seed: int = 0) -> float:
    """Two-sided sign-flip permutation p for a zero mean paired difference.

    The exact null for paired data: under H0 each round's difference is equally
    likely to have carried the opposite sign, so flipping signs at random
    generates the null distribution of the mean. Unlike `bootstrap_ci` this does
    not depend on a percentile of a resampled distribution, so it does not have
    the boundary instability that motivated `verdict_is_fragile`.
    """
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(d.mean()))
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(d))) * d).mean(axis=1)
    return float((np.abs(null) >= observed).mean())


def verdict_is_fragile(differences: np.ndarray, n_boot: int,
                       n_seeds: int = 4) -> bool:
    """Does the CI's significance verdict survive a different bootstrap seed?

    `bootstrap_ci` fixes its seed so the same data always prints the same
    interval. That is reproducible but not *stable*: the percentile bound
    carries its own Monte-Carlo error, and when the truth sits within it the
    verdict is a coin flip that looks deterministic. Audit 10 found E39's
    headline "+0.116 [+0.002, +0.233]" excluded zero on a minority of seeds,
    and E40 hit six such rows on fresh data. Re-draw under other seeds and
    report disagreement rather than hiding it.
    """
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        return False
    verdicts = set()
    for seed in range(12345, 12345 + n_seeds + 1):
        _, low, high = bootstrap_ci(d, n_boot, rng=np.random.default_rng(seed))
        verdicts.add((low > 0) or (high < 0))
    return len(verdicts) > 1


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

def summarise(path: Path, metrics: list[str], markdown: bool, n_boot: int,
              latex: bool = False) -> None:
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

    if latex:
        table_rows = []
        for agent, agent_rows in by_agent.items():
            cells = [tex_code(agent)]
            for metric in metrics:
                column, _, _, fmt = METRICS[metric]
                mean, low, high = bootstrap_ci(
                    np.array([r[column] for r in agent_rows]), n_boot)
                cells.append(tex_ci(fmt, mean, low, high))
            table_rows.append(cells)
        scenario = meta.get("scenario", "classic")
        tex_table(
            caption=(f"Performance over {n_rounds} rounds "
                     f"(scenario \\texttt{{{tex_escape(scenario)}}}, "
                     f"commit \\texttt{{{tex_escape(meta.get('git_commit','?'))}}}). "
                     f"Each cell is the mean with its 95\\,\\% bootstrap "
                     f"confidence interval in brackets."),
            label=f"tab:summary-{path.stem.replace('_', '-')}",
            column_spec="l" + "r" * len(metrics),
            header=["Agent"] + [METRICS[m][1] for m in metrics],
            rows=table_rows,
        )
    elif markdown:
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

    # The CI still decides the verdict -- that is the rule in README.md and
    # changing it here would silently reclassify every number already reported.
    # These two only *flag* a verdict that rests on the boundary.
    standard_error = float(differences.std(ddof=1) / np.sqrt(len(differences))) \
        if len(differences) > 1 else float("nan")
    t_stat = float(mean_diff / standard_error) if standard_error else float("nan")
    p_flip = signflip_p(differences)
    fragile = (significant != (p_flip < 0.05)) or verdict_is_fragile(differences, n_boot)

    return {
        "metric": metric, "label": label, "fmt": fmt,
        "a": float(values_a.mean()), "b": float(values_b.mean()),
        "diff": mean_diff, "low": low, "high": high,
        "verdict": verdict,
        "t": t_stat, "p_flip": p_flip, "fragile": fragile,
        "p": wilcoxon_p(differences),
        "win": float(np.mean(values_b > values_a)),
        "n": len(differences),
    }


def compare(path_a: Path, path_b: Path, agent: str | None, metrics: list[str],
            markdown: bool, n_boot: int, plot: Path | None = None,
            latex: bool = False) -> None:
    rows_a = rows_for(load(path_a), agent)
    rows_b = rows_for(load(path_b), agent)
    name_a, name_b = path_a.stem, path_b.stem

    lines = [paired_effect(rows_a, rows_b, metric, n_boot) for metric in metrics]

    n_paired = lines[0]["n"]
    agent_a, agent_b = rows_a[0]["agent"], rows_b[0]["agent"]
    who = agent_a if agent_a == agent_b else f"{agent_a} -> {agent_b}"
    title = f"{name_b}  vs  {name_a}   (agent: {who}, {n_paired} paired rounds)"

    if latex:
        tex_table(
            caption=(f"Paired comparison of \\texttt{{{tex_escape(name_b)}}} "
                     f"against \\texttt{{{tex_escape(name_a)}}} over "
                     f"{n_paired} rounds on identical arenas. The difference "
                     f"column is the mean of the per-round difference "
                     f"($B-A$) with its 95\\,\\% bootstrap confidence "
                     f"interval; an interval containing zero means the effect "
                     f"is not demonstrated at this sample size."
                     + ("  Rows marked $\\dagger$ sit on the significance "
                        "boundary: the interval and a sign-flip permutation "
                        "test disagree, or the interval's verdict changes under "
                        "a different bootstrap seed."
                        if any(l["fragile"] for l in lines) else "")),
            label=f"tab:compare-{name_a.replace('_','-')}-{name_b.replace('_','-')}",
            column_spec="lrrrl",
            header=["Metric", f"A: {tex_code(name_a)}", f"B: {tex_code(name_b)}",
                    "Difference (95\\,\\% CI)", "Verdict"],
            rows=[[line["label"],
                   f"${line['fmt'].format(line['a'])}$",
                   f"${line['fmt'].format(line['b'])}$",
                   tex_ci(line["fmt"], line["diff"], line["low"], line["high"],
                          signed=True),
                   tex_escape(line["verdict"]) + ("$^\\dagger$" if line["fragile"] else "")]
                  for line in lines],
        )
    elif markdown:
        print(f"\n**{title}**\n")
        print("| Metric | " + f"{name_a} | {name_b} | Difference (95 % CI) | "
              "t | sign-flip p | Verdict |")
        print("|---|---|---|---|---|---|---|")
        for line in lines:
            fmt = line["fmt"]
            mark = " **(fragile)**" if line["fragile"] else ""
            print(f"| {line['label']} | {fmt.format(line['a'])} | "
                  f"{fmt.format(line['b'])} | "
                  f"{line['diff']:+.3f} [{line['low']:+.3f}, {line['high']:+.3f}] | "
                  f"{line['t']:+.2f} | {line['p_flip']:.4f} | "
                  f"{line['verdict']}{mark} |")
        print("\n<sub>Paired difference (B − A) on identical arenas, "
              "95 % bootstrap CI. A CI containing 0 means the change is not "
              "demonstrated at this sample size. <b>(fragile)</b> means the CI "
              "and the sign-flip p disagree, or the CI's verdict changes under a "
              "different bootstrap seed -- the effect sits on the significance "
              "boundary and the verdict should not be quoted without a larger n.</sub>")
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
            extra = (f"    B better in {line['win']:.0%} of rounds, "
                     f"t={line['t']:+.2f}, sign-flip p={line['p_flip']:.4g}")
            if line["p"] is not None:
                extra += f", Wilcoxon p={line['p']:.4g}"
            if line["fragile"]:
                extra += "\n    ^ FRAGILE: on the significance boundary -- the CI and the "
                extra += "sign-flip p disagree, or the CI flips under another bootstrap seed"
            print(extra)
        print()

    if plot is not None:
        plot_compare(lines, name_a, name_b, n_paired, plot)


def _as_contribution(effect: dict, metric: str) -> dict:
    """Turn a raw ``variant − baseline`` difference into the *contribution of the
    removed component*, oriented so that positive always means "it helped".

    In a removal ablation the baseline is the **full** agent and each variant has
    one component taken out. The raw difference then answers "how did the crippled
    agent do", which reads backwards: dropping reward shaping lowers the score, so
    the tool would print WORSE -- while what we mean is "shaping matters".

    So we negate for metrics where more is better, and leave the sign alone for
    metrics where less is better:

        score:    removal lowers it   -> diff < 0 -> contribution = −diff > 0
        suicides: removal raises them -> diff > 0 -> contribution = +diff > 0

    Either way a positive number means the component improves the metric by that
    much, and the verdict can talk about the component instead of the cripple.
    """
    higher_better = METRICS[metric][2]
    sign = -1.0 if higher_better else 1.0

    low, high = sorted([sign * effect["low"], sign * effect["high"]])
    contribution = sign * effect["diff"]

    if low <= 0 <= high:
        verdict = "no effect shown"
    elif contribution > 0:
        verdict = "MATTERS"          # removing it made the agent worse
    else:
        verdict = "HARMFUL"          # removing it made the agent better

    return {**effect, "diff": contribution, "low": low, "high": high,
            "verdict": verdict}


def ablation(base: Path, variants: list[Path], metric: str, agent: str | None,
             n_boot: int, markdown: bool, plot: Path | None,
             mode: str = "removal", latex: bool = False) -> None:
    """One metric, many variants, all measured against the same baseline.

    ``mode="removal"`` -- the classic ablation. Baseline is the full agent, each
    variant has exactly one component removed, and the figure shows what that
    component contributes.

    ``mode="addition"`` -- baseline is the minimal agent and each variant adds one
    component. Raw differences are shown as-is.
    """
    rows_base = rows_for(load(base), agent)
    entries = []
    for path in variants:
        effect = paired_effect(rows_base, rows_for(load(path), agent), metric, n_boot)
        if mode == "removal":
            effect = _as_contribution(effect, metric)
        effect["name"] = path.stem
        entries.append(effect)

    label = METRICS[metric][1]
    removal = mode == "removal"
    title = (f"{label}: contribution of each component (removed from {base.stem})"
             if removal else f"{label}: variants vs {base.stem}")
    value_header = "Contribution (95 % CI)" if removal else "Difference vs baseline (95 % CI)"
    column_header = "Component removed" if removal else "Variant"
    axis = (f"contribution to {label}   (positive = component helps)"
            if removal else f"paired difference in {label}")

    if latex:
        anchor_value = entries[0]["fmt"].format(entries[0]["a"])
        caption = (
            (f"Ablation study on \\emph{{{label}}}: each row reports what one "
             f"component contributes, measured by removing it from the full "
             f"agent \\texttt{{{tex_escape(base.stem)}}} "
             f"(${anchor_value}$). A positive value means removing the "
             f"component made the agent worse, i.e.\\ it earns its place. "
             if removal else
             f"Variants against the baseline "
             f"\\texttt{{{tex_escape(base.stem)}}} (${anchor_value}$) "
             f"on \\emph{{{label}}}. ")
            + f"Paired over {entries[0]['n']} rounds on identical arenas, "
              f"95\\,\\% bootstrap confidence intervals."
        )
        tex_table(
            caption=caption,
            label=f"tab:ablation-{metric.replace('_','-')}",
            column_spec="lrrl",
            header=[column_header,
                    f"{label} without it" if removal else label,
                    value_header.replace("95 % CI", "95\\,\\% CI"),
                    "Verdict"],
            rows=[[tex_code(entry["name"]),
                   f"${entry['fmt'].format(entry['b'])}$",
                   tex_ci(entry["fmt"], entry["diff"], entry["low"],
                          entry["high"], signed=True),
                   tex_escape(entry["verdict"])]
                  for entry in entries],
        )
    elif markdown:
        print(f"\n**{title}** ({entries[0]['n']} paired rounds)\n")
        print(f"| {column_header} | {label} without it | {value_header} | Verdict |"
              if removal else
              f"| {column_header} | {label} | {value_header} | Verdict |")
        print("|---|---|---|---|")
        for entry in entries:
            fmt = entry["fmt"]
            print(f"| `{entry['name']}` | {fmt.format(entry['b'])} | "
                  f"{entry['diff']:+.3f} [{entry['low']:+.3f}, {entry['high']:+.3f}] | "
                  f"{entry['verdict']} |")
        note = (f"Full agent `{base.stem}`: {entries[0]['fmt'].format(entries[0]['a'])}. "
                f"Positive contribution = removing the component made the agent worse, "
                f"i.e. it earns its place. " if removal else
                f"Baseline `{base.stem}`: {entries[0]['fmt'].format(entries[0]['a'])}. ")
        print(f"\n<sub>{note}Paired on identical arenas, 95 % bootstrap CI.</sub>")
    else:
        print(f"\n{title}")
        print("=" * len(title))
        anchor = "full agent" if removal else "baseline"
        print(f"  {anchor} {base.stem}: {entries[0]['fmt'].format(entries[0]['a'])}"
              f"   ({entries[0]['n']} paired rounds)\n")
        width = max(len(e["name"]) for e in entries) + 2
        for entry in entries:
            ci = f"[{entry['low']:+.3f}, {entry['high']:+.3f}]"
            print(f"  {entry['name']:<{width}}{entry['diff']:>+9.3f}{ci:>22}   "
                  f"{entry['verdict']}")
        if removal:
            print("\n  MATTERS = removing it made the agent worse, so it earns its place."
                  "\n  HARMFUL = the agent was better without it.")
        print()

    if plot is not None:
        plot_forest(entries, title, axis, plot,
                    positive_is_better=True if removal else None)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
VERDICT_COLOURS = {
    "BETTER": "tab:green",
    "WORSE": "tab:red",
    "no effect shown": "tab:gray",
    # removal ablation: the verdict is about the component, not the crippled agent
    "MATTERS": "tab:green",
    "HARMFUL": "tab:red",
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


def plot_compare(lines: list[dict], name_a: str, name_b: str, n_paired: int,
                 out: Path) -> Path:
    """One small panel per metric for a paired A/B comparison.

    Deliberately *not* a single shared axis: the metrics have different units
    (score in points, survival as a rate, think time in ms), so putting them on
    one x-axis squashes everything small onto the zero line. Each panel gets its
    own scale instead.

    Reading a panel: the dot is the mean paired difference, the whiskers its
    95 % CI, the dashed line is "no change". The green band marks the improving
    direction -- which differs per metric, since fewer suicides is good but more
    coins is good. Whiskers overlapping the dashed line = effect not shown.
    """
    plt = _pyplot()

    n = len(lines)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows + 0.8),
                             squeeze=False)

    for index, line in enumerate(lines):
        ax = axes[index // cols][index % cols]
        higher_better = METRICS[line["metric"]][2]
        fmt = line["fmt"]
        diff, low, high = line["diff"], line["low"], line["high"]

        span = max(abs(low), abs(high), abs(diff), 1e-9)
        left, right = min(low, 0) - 0.45 * span, max(high, 0) + 0.45 * span

        # Shade the half of the axis that means "better" for THIS metric.
        if higher_better:
            ax.axvspan(0, right, color="tab:green", alpha=0.08, zorder=0)
        else:
            ax.axvspan(left, 0, color="tab:green", alpha=0.08, zorder=0)

        ax.axvline(0, color="black", linestyle="--", linewidth=1.1, zorder=2)
        colour = VERDICT_COLOURS[line["verdict"]]
        ax.errorbar(diff, 0, xerr=[[diff - low], [high - diff]], fmt="o",
                    color=colour, capsize=6, markersize=10, linewidth=2.4,
                    zorder=3)

        ax.set_xlim(left, right)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(line["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel(
            f"mean of per-round (B − A)   "
            f"({'higher' if higher_better else 'lower'} is better)",
            fontsize=8)

        # The numbers behind the dot, so each panel stands on its own.
        ax.text(0.5, 0.93,
                f"A {fmt.format(line['a'])}   →   B {fmt.format(line['b'])}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.text(0.5, 0.10,
                f"{diff:+.3f}  [{low:+.3f}, {high:+.3f}]\n{line['verdict']}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color=colour, fontweight="bold")

    for index in range(n, rows * cols):
        axes[index // cols][index % cols].axis("off")

    # Two short lines only. What is plotted is a mean of per-round differences,
    # not two CIs subtracted -- but that belongs in the report caption, not on
    # every figure. Keeping the header readable matters more.
    fig.text(0.5, 0.985, f"A: {name_a}    →    B: {name_b}",
             ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.945,
             f"{n_paired} paired rounds · mean per-round difference (B − A) "
             f"with 95 % CI",
             ha="center", va="top", fontsize=9, color="dimgray")

    top = 0.93 if rows > 1 else 0.86
    fig.tight_layout(rect=(0, 0, 1, top))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_forest(entries: list[dict], title: str, xlabel: str, out: Path,
                positive_is_better: bool | None = None) -> Path:
    """Forest plot: one row per entry, point estimate with 95 % CI, zero line.

    Only used for ablations, where every row shows the *same* metric -- the one
    case where a shared x-axis is meaningful.
    """
    plt = _pyplot()

    entries = list(entries)[::-1]        # first entry ends up on top
    positions = np.arange(len(entries))
    diffs = np.array([e["diff"] for e in entries])
    lower = diffs - np.array([e["low"] for e in entries])
    upper = np.array([e["high"] for e in entries]) - diffs
    colours = [VERDICT_COLOURS[e["verdict"]] for e in entries]

    height = max(3.0, 0.62 * len(entries) + 2.2)
    fig, ax = plt.subplots(figsize=(8.5, height))

    # Shade the improving half. All rows share one metric here, so "better" is
    # a single direction -- without this, green dots on both sides of zero are
    # genuinely confusing for metrics where less is more (suicides, invalid).
    if positive_is_better is None:
        higher_better = METRICS[entries[0]["metric"]][2] if entries else True
    else:
        # Values are already oriented (removal ablation): positive = helps.
        higher_better = positive_is_better
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1, zorder=2)
    for pos, diff, low, high, colour in zip(positions, diffs, lower, upper, colours):
        ax.errorbar(diff, pos, xerr=[[low], [high]], fmt="o", color=colour,
                    capsize=5, markersize=8, linewidth=2.0, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels([e["name"] for e in entries])
    ax.set_xlabel(xlabel if positive_is_better is not None else
                  f"{xlabel}   ({'higher' if higher_better else 'lower'} is better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.6, len(entries) - 0.4)

    # Breathing room, so a point at the extreme does not sit on the frame.
    lo = min(np.min(diffs - lower), 0.0)
    hi = max(np.max(diffs + upper), 0.0)
    pad = 0.18 * max(hi - lo, 1e-9)
    left, right = lo - pad, hi + pad
    ax.set_xlim(left, right)

    if higher_better:
        ax.axvspan(0, right, color="tab:green", alpha=0.07, zorder=0)
    else:
        ax.axvspan(left, 0, color="tab:green", alpha=0.07, zorder=0)
    ax.set_xlim(left, right)

    # Numbers directly above each point -- the figure should be readable on its
    # own, without cross-referencing the table.
    for pos, diff, low, high in zip(positions, diffs, lower, upper):
        ax.annotate(f"{diff:+.2f} [{diff - low:+.2f}, {diff + high:+.2f}]",
                    xy=(diff, pos), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="dimgray")

    # Only the verdicts this mode can produce -- showing BETTER/WORSE next to
    # MATTERS/HARMFUL would just raise questions.
    shown = (["MATTERS", "HARMFUL", "no effect shown"]
             if positive_is_better is not None else
             ["BETTER", "WORSE", "no effect shown"])
    handles = [plt.Line2D([], [], color=VERDICT_COLOURS[v], marker="o",
                          linestyle="", label=v) for v in shown]
    # Legend outside the plotting area: with few rows there is no free corner.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=8, frameon=False)

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


PRIMARY_COLOUR = "tab:green"      # the agent under test
OTHER_COLOUR = "lightsteelblue"   # opponents, deliberately muted


def primary_agent(rows: list[dict], agent: str | None) -> str | None:
    """The agent the run is about: the explicit ``--agent``, else slot 0.

    ``evaluate.py`` always puts ours in slot 0, so this picks it out without
    anyone having to name it.
    """
    if agent:
        return agent
    return next((r["agent"] for r in rows if r["slot"] == 0), None)


def plot_summary(path: Path, metrics: list[str], agent: str | None,
                 n_boot: int, out: Path) -> Path:
    """Bar chart per metric, one bar per agent, error bars = 95 % bootstrap CI.

    The agent under test is green, everyone else muted blue -- so it is obvious
    at a glance which bar the figure is actually about.
    """
    plt = _pyplot()

    rows = load(path)
    highlight = primary_agent(rows, agent)
    if agent:
        rows = [r for r in rows if r["agent"] == agent]
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent"]].append(row)
    names = list(by_agent)
    colours = [PRIMARY_COLOUR if name == highlight else OTHER_COLOUR
               for name in names]

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
               color=colours, alpha=0.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        # bold tick label too, so the highlight survives greyscale printing
        for tick, name in zip(ax.get_xticklabels(), names):
            if name == highlight:
                tick.set_fontweight("bold")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    for index in range(n, plot_rows * cols):       #  blank out unused panels
        axes[index // cols][index % cols].axis("off")

    if highlight is not None and len(names) > 1:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=PRIMARY_COLOUR,
                          label=f"{highlight} (tested)"),
            plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOUR, label="opponents"),
        ]
        axes[0][0].legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)

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
                        help="Ablation study on a single metric (see --metric). "
                             "First file = the full agent, the rest = one "
                             "component removed each.")
    parser.add_argument("--ablation-mode", default="removal",
                        choices=["removal", "addition"],
                        help="removal (default): baseline is the FULL agent and "
                             "each variant has one component taken out; the "
                             "figure shows what that component contributes. "
                             "addition: baseline is the minimal agent and each "
                             "variant adds one component.")
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
                        help="Emit a Markdown table.")
    parser.add_argument("--latex", action="store_true",
                        help="Emit a booktabs LaTeX table ready to paste into the report. Needs only the booktabs package.")
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
                plot=figure_path(f"compare_{base.stem}_vs_{new.stem}"),
                latex=args.latex)

    if args.ablation:
        base, *variants = args.ablation
        ablation(base, variants, args.metric, args.agent, args.n_boot,
                 args.markdown, plot=figure_path(f"ablation_{args.metric}"),
                 mode=args.ablation_mode, latex=args.latex)

    for path in args.files:
        summarise(path, args.metrics, args.markdown, args.n_boot,
                  latex=args.latex)
        target = figure_path(f"summary_{path.stem}")
        if target is not None:
            plot_summary(path, args.metrics, args.agent, args.n_boot, target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

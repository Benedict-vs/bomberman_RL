"""Checkpoint curves: a metric against training episodes, one line per arm.

The shape that has decided four experiments in a row and that neither
`analyze.py` nor `trainlog.py` draws. `analyze.py` compares two finished tables;
`trainlog.py` plots the *training* log, which E26 showed can disagree completely
with an eps = 0 evaluation (18.8 crates/episode logged, 3.74 measured). This
plots the evaluated checkpoints — the only numbers this project treats as real.

Each arm is a filename pattern containing `{seed}` and `{ep}`; every combination
that exists on disk is loaded, and the band is the t-interval over *training
runs* (n = seeds), which is the unit of analysis `MEASUREMENT.md` mandates from
rung 3 on — not a per-round bootstrap.

    uv run python tools/curves.py --out results/figs/e34.png \\
        --metric score won \\
        --seeds 100 101 102 103 104 \\
        --arm "E34 forced start" "benedict_q_e34_W_s{seed}__ep{ep}__task4_rb_val550731" \\
        --arm "E33 control"      "benedict_q_e33_ctl_s{seed}__ep{ep}__task4_rb_val550731" \\
        --hline "ceiling (rule, untrained)" 4.399 0.442 \\
        --hline "rule_based_agent"          3.254 0.286

`--hline` takes one value per `--metric`, in the same order.
"""

import argparse
import csv
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "results" / "eval" / "task4_tournament"

# metric -> (csv column, axis label, higher_is_better)
METRICS = {
    "score": ("score", "Score", True),
    "won": ("won", "Win rate", True),
    "suicides": ("suicides", "Suicides / round", False),
    "survived": ("survived", "Survival rate", True),
    "crates": ("crates", "Crates / round", True),
    "kills": ("kills", "Kills / round", True),
    "coins": ("coins", "Coins / round", True),
    "invalid": ("invalid", "Invalid actions", False),
}


def agent_mean(path: Path, column: str, prefix: str) -> float | None:
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r["agent"].startswith(prefix)]
    if not rows:
        return None
    return sum(float(r[column]) for r in rows) / len(rows)


def t95(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of the 95 % t-interval over training runs."""
    mean = st.mean(values)
    if len(values) < 2:
        return mean, 0.0
    # t(0.975, df) for the small n this project uses; falls back to 1.96.
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    t = table.get(len(values) - 1, 1.96)
    return mean, t * st.stdev(values) / len(values) ** 0.5


def collect(pattern: str, seeds: list[str], eps: list[int], column: str,
            prefix: str) -> tuple[list[int], list[float], list[float]]:
    xs, ys, es = [], [], []
    for ep in eps:
        vals = [v for s in seeds
                if (v := agent_mean(EVAL / f"{pattern.format(seed=s, ep=ep)}.csv",
                                    column, prefix)) is not None]
        if not vals:
            continue
        mean, half = t95(vals)
        xs.append(ep)
        ys.append(mean)
        es.append(half)
    return xs, ys, es


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", metavar=("LABEL", "PATTERN"),
                    required=True, help="Repeatable. PATTERN uses {seed} and {ep}.")
    ap.add_argument("--metric", nargs="+", default=["score", "won"],
                    choices=sorted(METRICS))
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--eps", nargs="+", type=int,
                    default=[500, 2000, 5000, 10000, 20000])
    ap.add_argument("--hline", nargs="+", action="append", metavar="LABEL VALUE...",
                    help="Reference line: label then one value per --metric.")
    ap.add_argument("--prefix", default="benedict",
                    help="Agent-name prefix identifying our rows in the CSV.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(args.metric)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for j, metric in enumerate(args.metric):
        column, label, higher = METRICS[metric]
        ax = axes[0][j]

        for i, (arm_label, pattern) in enumerate(args.arm):
            xs, ys, es = collect(pattern, args.seeds, args.eps, column, args.prefix)
            if not xs:
                print(f"  (no files for arm {arm_label!r} — check the pattern)")
                continue
            c = colours[i % len(colours)]
            ax.plot(xs, ys, marker="o", color=c, linewidth=1.8, label=arm_label)
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)], color=c, alpha=0.15)

        for k, spec in enumerate(args.hline or []):
            name, values = spec[0], [float(v) for v in spec[1:]]
            if j < len(values):
                ax.axhline(values[j], linestyle="--", linewidth=1.2,
                           color=f"C{7 + k}", alpha=0.8, label=name)

        ax.set_xlabel("training episodes")
        ax.set_ylabel(label + ("" if higher else "  (lower is better)"))
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(args.title or "Evaluated checkpoints, 1000 rounds at eps = 0"
                 "  ·  band = 95 % t-interval over training runs")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

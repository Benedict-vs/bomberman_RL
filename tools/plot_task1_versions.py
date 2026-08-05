"""One-off: task-1 comparison of the model versions (figure labelled in English).

Important: the round ends as soon as all 50 coins are collected. `steps` is
therefore only a path-length measure for *completed* rounds -- 400 means "never
finished", not "slow". Hence three panels: coins, completion rate, and path
length over the completed rounds only.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/maxi/Projects/bomberman_RL")
sys.path.insert(0, str(REPO))
from tools.analyze import load, rows_for, bootstrap_ci  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = REPO / "results/eval"
RUNS = [
    ("v1\noriginal",              E / "maxi_q_v1__task1.csv"),
    ("v2\nmask + decay",          E / "maxi_q_v2__task1.csv"),
    ("v3\n+ step cost",           E / "maxi_q_v3__task1.csv"),
    ("reference\ncoin_collector", E / "baselines/baseline_coin_collector_agent__task1.csv"),
]
COLORS = ["#9e9e9e", "#c0504d", "#2e7d32", "#1f77b4"]

data = []
for name, path in RUNS:
    rows = rows_for(load(path), None)
    coins = np.array([float(r["coins"]) for r in rows])
    steps = np.array([float(r["round_steps"]) for r in rows])
    done = coins >= 50
    data.append({"name": name, "coins": coins, "done": done,
                 "steps_done": steps[done]})


def bars(ax, values, ylabel, title, fmt="{:.1f}", ref=None, ref_label=None):
    xs, ms, lo, hi, cols, names = [], [], [], [], [], []
    for i, (d, c) in enumerate(zip(data, COLORS)):
        v = values(d)
        names.append(d["name"])
        if v is None or len(v) == 0:
            xs.append(i); ms.append(0); lo.append(0); hi.append(0); cols.append(c)
            continue
        m, l, h = bootstrap_ci(np.asarray(v, dtype=float))
        xs.append(i); ms.append(m); lo.append(max(m - l, 0)); hi.append(max(h - m, 0)); cols.append(c)
    ax.bar(xs, ms, color=cols, width=0.62, yerr=[lo, hi], capsize=5, ecolor="#333", zorder=3)
    for x, m, d in zip(xs, ms, data):
        v = values(d)
        label = "no\nrounds" if v is None or len(v) == 0 else fmt.format(m)
        ax.text(x, m, label, ha="center", va="bottom", fontsize=9.5, zorder=4)
    if ref is not None:
        ax.axhline(ref, ls="--", lw=1, color="#1f77b4", zorder=2)
        if ref_label:
            ax.text(0.01, ref, " " + ref_label, color="#1f77b4", va="bottom", ha="left",
                    fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    # Kopfraum, damit Balkenbeschriftung und Referenzlinien-Label sich nicht überlagern
    top = max([m + h for m, h in zip(ms, hi)] + ([ref] if ref is not None else []))
    ax.set_ylim(top=top * 1.16)


fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
bars(axes[0], lambda d: d["coins"], "Coins per round",
     "Coins - maximum 50", ref=50, ref_label="maximum 50")
bars(axes[1], lambda d: d["done"].astype(float) * 100, "Rounds with all 50 coins [%]",
     "Completion rate", fmt="{:.0f} %", ref=100)
bars(axes[2], lambda d: d["steps_done"], "Steps to the last coin",
     "Path length - completed rounds only\n(lower is better)",
     ref=125.3)  # Linie braucht keine Beschriftung, der Referenzbalken steht daneben

n_done = " · ".join(f"{d['name'].splitlines()[0]}: n={int(d['done'].sum())}" for d in data)
fig.suptitle("maxi_coin_collector, task 1 (coin-heaven), 300 rounds, seed 20260731, eps = 0\n"
             f"completed rounds - {n_done}", fontsize=10.5)
fig.tight_layout()
out = REPO / "results/figures/maxi_task1_steps_by_version.png"
fig.savefig(out, dpi=200)
print("Wrote", out)
for d in data:
    n = int(d["done"].sum())
    s = f"{d['steps_done'].mean():.1f}" if n else "--"
    print(f"  {d['name'].replace(chr(10), ' '):28s} coins={d['coins'].mean():5.2f}  "
          f"completed={n:3d}/300  steps(completed)={s}")

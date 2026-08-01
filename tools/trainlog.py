#!/usr/bin/env python3
"""One-episode-per-row training log, so all three of us produce comparable
learning curves.

Evaluation (``tools/evaluate.py``) answers "how good is the finished agent".
This answers "how did it get there" -- the training-progress diagrams the
report asks for.

Use from ``train.py``
---------------------
::

    from tools.trainlog import TrainLogger          # see note below

    def setup_training(self):
        self.trainlog = TrainLogger(
            agent=self.__class__.__module__.split(".")[0],
            run="q_v3_task2",
            hyperparams={"alpha": 0.1, "gamma": 0.95, "eps_decay": 0.9995},
        )

    def end_of_round(self, last_game_state, last_action, events):
        ...
        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=last_game_state["self"][1],
            steps=last_game_state["step"],
            events=self.episode_events,        # list[str] you collected
            reward=self.episode_reward,        # shaped reward total
            epsilon=self.epsilon,
            extra={"td_error": float(np.mean(self.losses))},
        )

Import safety
-------------
``tools/`` is **not** part of the submission -- only ``agent_code/<name>/`` is
zipped up. That is fine, because ``train.py`` is only imported when the game
runs with ``--train``, and the tournament never does. Still, import defensively
so a stray import can never crash a tournament game::

    try:
        from tools.trainlog import TrainLogger
    except ImportError:
        TrainLogger = None

Output
------
``results/train/<agent>__<run>.csv`` -- appended to, so an interrupted run that
you restart does not lose its history. A sidecar ``.meta.json`` records the
hyperparameters, which is what makes a curve interpretable three weeks later.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Events worth a dedicated column: these are the ones we steer on.
TRACKED_EVENTS = [
    "COIN_COLLECTED",
    "CRATE_DESTROYED",
    "COIN_FOUND",
    "KILLED_OPPONENT",
    "KILLED_SELF",
    "GOT_KILLED",
    "SURVIVED_ROUND",
    "INVALID_ACTION",
    "BOMB_DROPPED",
    "WAITED",
]

BASE_COLUMNS = [
    "episode", "wall_clock_s", "score", "steps", "reward", "epsilon",
]


class TrainLogger:
    """Append-only CSV writer for per-episode training metrics."""

    def __init__(
        self,
        agent: str,
        run: str,
        hyperparams: dict | None = None,
        out_dir: Path | str | None = None,
        extra_columns: list[str] | None = None,
        flush_every: int = 20,
    ):
        self.agent = agent
        self.run = run
        self.extra_columns = list(extra_columns or [])
        self.flush_every = flush_every
        self._since_flush = 0
        self._started = time.time()

        directory = Path(out_dir) if out_dir else REPO_ROOT / "results" / "train"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{agent}__{run}.csv"
        self.meta_path = directory / f"{agent}__{run}.meta.json"

        self.columns = BASE_COLUMNS + TRACKED_EVENTS + self.extra_columns
        is_new = not self.path.exists()
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns,
                                      extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

        self._write_meta(hyperparams or {})

    # -- meta ------------------------------------------------------------
    def _write_meta(self, hyperparams: dict) -> None:
        meta = {
            "agent": self.agent,
            "run": self.run,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "hyperparams": hyperparams,
            "columns": self.columns,
        }
        history = []
        if self.meta_path.exists():
            try:
                with open(self.meta_path) as fh:
                    previous = json.load(fh)
                history = previous.get("history", [])
                history.append({k: previous[k] for k in
                                ("started_utc", "git_commit", "hyperparams")
                                if k in previous})
            except (json.JSONDecodeError, OSError):
                pass
        meta["history"] = history
        with open(self.meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

    # -- logging ---------------------------------------------------------
    def log_episode(
        self,
        episode: int,
        score: float,
        steps: int,
        events: list[str] | None = None,
        reward: float = 0.0,
        epsilon: float = float("nan"),
        extra: dict | None = None,
    ) -> None:
        """Record one finished episode. Call this from ``end_of_round``."""
        counts = Counter(events or [])
        row = {
            "episode": episode,
            "wall_clock_s": round(time.time() - self._started, 2),
            "score": score,
            "steps": steps,
            "reward": round(float(reward), 4),
            "epsilon": round(float(epsilon), 6),
        }
        row.update({event: counts.get(event, 0) for event in TRACKED_EVENTS})
        row.update(extra or {})

        self._writer.writerow(row)
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_flush = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_curves(paths, metric="score", window=100, out=None, labels=None):
    """Smoothed learning curves from one or more training logs.

    ``python tools/trainlog.py results/train/*.csv --metric score``
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, path in enumerate(paths):
        path = Path(path)
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        values = np.array([float(r[metric]) for r in rows])
        episodes = np.array([float(r["episode"]) for r in rows])

        if len(values) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(values, kernel, mode="valid")
            centres = episodes[window - 1:]
        else:
            smoothed, centres = values, episodes

        label = labels[index] if labels else path.stem
        ax.plot(centres, smoothed, label=label, linewidth=1.6)
        ax.plot(episodes, values, alpha=0.12, linewidth=0.6,
                color=ax.lines[-1].get_color())

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{metric} (moving average, window={window})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(out) if out else REPO_ROOT / "results" / "train" / f"curve_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training curves.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--metric", default="score")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--out", default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()
    plot_curves(args.files, args.metric, args.window, args.out, args.labels)

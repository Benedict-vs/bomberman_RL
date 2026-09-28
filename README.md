# Reinforcement Learning for Bomberman

Final project for Machine Learning Essentials (SoSe 2026, Heidelberg University) by
Benedict von Schubert, Ben Gschwend and Maximilian Bohn.

We trained agents for the four-player Bomberman framework from
[ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl), which this repository is a
fork of. The report is handed in separately and is not part of this repository.

## The two agents

| | folder | method | status |
|---|---|---|---|
| Agent A | `agent_code/Arminator/` | tabular Q-learning on a hand-built feature map | submitted to the tournament |
| Agent B | `agent_code/Agent_B/` | deep Q-network on spatial board channels | second model, described in the report |

Against three `rule_based_agent` opponents (1000 rounds, `classic`), Agent A scores 4.015 points
per round and Agent B 3.915, while `rule_based_agent` itself scores 3.254 in the same slot.
Details and the comparison between the two are in the report.

`agent_code/Arminator/README.md` describes Agent A's features, its training command and the
measured think time.

## Requirements

- Agent A (submitted): NumPy only.
- Agent B: NumPy and PyTorch (>= 1.13, CPU only).
- Training and analysis scripts: SciPy, matplotlib, tqdm, pygame for the GUI.

All of these are in the provided `Dockerfile`. For local work we used Python 3.12 with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Running

Watch Agent A against three rule-based agents:

```bash
uv run python main.py play --agents Arminator rule_based_agent rule_based_agent rule_based_agent
```

Evaluate an agent over many rounds without the GUI. The script writes one CSV row per round and
agent to `results/eval/`, which `analyze.py` turns into means with bootstrap confidence intervals
or paired comparisons:

```bash
uv run python tools/evaluate.py --agents Arminator --opponents rule_based --n-rounds 300 --label my_run
uv run python tools/analyze.py results/eval/my_run.csv
uv run python tools/analyze.py --compare results/eval/old.csv results/eval/new.csv
```

`python main.py play --help` lists the framework's own options.

## Repository layout

- `agent_code/`: our agents next to the provided ones (`rule_based_agent`, `coin_collector_agent`,
  `peaceful_agent`, `random_agent`, `tpl_agent`, `user_agent`).
- `tools/`: the evaluation and analysis scripts shared by both models (`evaluate.py`,
  `analyze.py`, `trainlog.py`, plotting). Not part of the submission.
- `experiments/`: our experiment ledgers. Each entry records the question, the change, the
  prediction written before the run, the commit and the result.
- `results/eval/`: evaluation CSVs with `.meta.json` files (commit, seed, rule settings), grouped
  by task. `results/train/` holds training logs in the same grouping.
- `checkpoints/`: the two Q-tables Agent A is warm-started from, needed to reproduce its training.
- `BENEDICT.md`, `BEN.md`, `MAXI.md`: personal logbooks with decisions and the reasoning behind
  them. `KONZEPT.md` is the initial project plan (German).
- `MEASUREMENT.md`: why the measurement rules below are what they are.

The framework files (`main.py`, `environment.py`, `agents.py`, `items.py`, `settings.py`) are
unchanged from upstream except for one addition to `settings.py`: setting the environment variable
`BM_QUIET_LOGS=1` lowers the log level during long training runs. Without it the values are the
upstream ones.

## Task ladder

We trained along four tasks of increasing difficulty. The `--opponents` presets of
`evaluate.py` follow it.

1. `coin-heaven`, no crates or opponents: collect coins efficiently.
2. `classic` without opponents: open crates with bombs and escape your own bombs.
3. `classic` with `peaceful_agent` or `coin_collector_agent`: survive and hunt.
4. `classic` with `rule_based_agent`: the tournament setting.

## Measurement conventions

We agreed on these so that numbers from different people stay comparable:

- Every evaluation uses the same base seed (20260731). `evaluate.py` reseeds each round with
  `base_seed + round_index`, so round *i* is the same arena for every agent. That is what makes
  the comparisons paired.
- 300 rounds for a reported number, 100 for a quick check, 1000 for final measurements.
  Files are named `<person>_<model>_<version>__<task>.csv`.
- A change counts as an improvement only if the paired 95 % confidence interval excludes 0 and
  `analyze.py` does not mark the row as `(fragile)`.
- A training curve is not a result. Agents are evaluated at epsilon = 0 with `evaluate.py`.
- The primary metric is the score per round, since the tournament is decided by total score.
- Training logs go to `results/train/<task>/`, next to the evaluations of the same task in
  `results/eval/<task>/`.

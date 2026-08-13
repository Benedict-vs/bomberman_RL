# Bomberman RL — Final Project (MLE, Uni Heidelberg, SS 2026)

Train an RL agent to play a 4-player Bomberman variant. Tournament + report.
Source of truth: `final_project.pdf`. Approach and milestones: `KONZEPT.md`.
This file is the terse spec — only what an agent cannot read out of the codebase.
Companion docs, read on demand rather than every session:
`MEASUREMENT.md` (why the measurement rules are what they are), `FRAMEWORK_README.md`.

## Working agreement (for any AI agent on this repo)
- **Never run `git commit`, `git push`, `git add`, or otherwise stage/commit changes.** The user
  commits, always. Edit files and report what changed; leave the working tree for them to review.
- Same for `.gitignore` and any other git plumbing — propose the change, don't apply it silently.
- Personal logbooks (`BENEDICT.md`, `MAXI.md`, `BEN.md`) are each member's diary: current state,
  ideas, and the reasoning behind decisions. Read the relevant one to pick up where someone left
  off; only append to a logbook when its owner asks, and never rewrite existing entries.

## Hard rules
- **The solution must involve machine learning** — a purely rule-based agent is rejected.
  The model must *learn from* the features; a feature that returns "the best action" is not allowed.
- **No multiprocessing in the final agent** (training may use anything, incl. GPUs — but the
  agent runs **CPU-only** during the tournament).
- Only the agent's own subdirectory of `agent_code/` is submitted; the framework is reset to the
  original version for official games, so **never rely on edits to `environment.py`/`settings.py`/etc.**
- **Relative paths only** inside agent code (absolute paths are the classic submission crash).
- At least **two different agents/models** must be developed and described in the report.
- Team-wide work, no per-member silos. AI drafts must be refined into our own style.
- No copy-pasting existing solutions; no resubmission of previous years' code.

## Deadlines
- Submission test (MaMPf, optional but recommended): **17.09.2026, 21:00**
- Agent code: **21.09.2026, 21:00** — zip as `final-project-agent-code.zip`
- Report (PDF, ~4000 words *per team member*, sections marked with their main author) + public
  repo URL: **28.09.2026, 21:00**. Do **not** upload the report to the repo.

## Game rules
Constants are in `settings.py`; these are the ones that shape feature design and are not
obvious from reading it:
- Explosions reach `BOMB_POWER = 3` tiles in 4 directions, are blocked by stone walls and do
  **not** turn corners; they linger `EXPLOSION_TIMER = 2` after `BOMB_TIMER = 4`.
- `field` uses **image coords (x,y)**, so it prints transposed vs. the GUI.
- Rewards: coin `+1`, kill `+5`. Killing opponents decides the score.
- `TIMEOUT = 0.5 s` per step (no limit while training). Exceeding it → `WAIT`, and the overrun is
  subtracted from the next step's budget. Reference hardware: one thread of an AMD Ryzen 5 2600,
  ≤8 GB RAM — so budget against a machine far slower than ours.
- Tournament scenario: `classic`. Dev scenarios: `empty`, `coin-heaven`, `loot-crate`.

## Agent interface — `agent_code/<name>/`
Signatures and `game_state` keys: read `agent_code/tpl_agent/`; event names: `events.py`.
`callbacks.py` is always loaded (`setup`, `act`); `train.py` only with `--train N`
(`setup_training`, `game_events_occurred`, `end_of_round`). `self.logger` and `self.train` are
preset. `act()` must return one of `'UP','DOWN','LEFT','RIGHT','BOMB','WAIT'` within 0.5 s.

**`GOT_KILLED` means "died", not "killed by an opponent".** `environment.py:264` adds it to
*every* agent removed by an explosion, and `environment.py:251` adds `KILLED_SELF` **in
addition** when the blast was its own. A reward table with both therefore prices a suicide at
the *sum* of the two and an opponent's kill at `GOT_KILLED` alone — the opposite of the
symmetry it looks like. Discovered the expensive way in `experiments/benedict.md` E25, where an
intended −5/−5 was really −10/−5. To price "death" once, put the whole penalty on `GOT_KILLED`
and leave `KILLED_SELF` at 0; that is *identical* to the rung-2 table on a board with no
opponents, since a suicide fires both events there too. Split them only when own-bomb and
enemy-bomb deaths are deliberately priced differently, and write down which is which.

## Running the game
`python main.py play --help` for the full CLI. Useful flags: `--no-gui` (fast training),
`--skip-frames`, `--seed`, `--turn-based` (with `user_agent`), `--continue-without-training`,
`--save-replay` (needed before `main.py replay <file>`).
Prefer `uv run python main.py …` so the project venv is used.

Three things that bite during long training sweeps (numbers in `MEASUREMENT.md`):
- **`BM_QUIET_LOGS=1`** drops all three log levels to WARNING — always set it for a sweep.
  It is an environment switch, not an edited constant, so there is nothing to restore before
  submitting.
- `agents.py:226-228` hardcodes the agent log path in mode `"w"`, so **parallel runs of the same
  agent overwrite each other's log** — during a sweep the training CSV is the only usable record.
- Check how often `train.py` calls `np.save`; once per round on a large table makes disk I/O the
  bottleneck of a multi-run sweep.

## Models we are building
- **Model A — tabular Q-learning on hand-built features.** Lecture technique, fast to converge,
  interpretable. The baseline everything is measured against and the tournament fallback.
- **Model B — DQN on the raw board** (7 channels × 17×17, small CNN). GPU for training, CPU-only
  inference in the tournament. Clear go/no-go: if it does not beat `rule_based_agent` in time,
  Model A is submitted. Model B goes into the report either way — a documented failure is a
  valid result.
- Feature extraction, reward scheme, evaluation and training logs are **shared** between both.
  Splitting the team per model is explicitly forbidden by the task description.

## Task ladder (subsets of each other)
1. `coin-heaven`, no crates/opponents → efficient navigation to revealed coins.
2. `classic`, no opponents → use bombs to open crates, **escape own bombs**, keep navigating.
3. Crates + `peaceful_agent` (easy) / `coin_collector_agent` (hard) → hunt and kill.
4. Crates + `rule_based_agent` → must beat it to have a shot at the tournament.

**Keep the action set in step with the features.** `BOMB` in the action space without a danger
feature ("am I in a blast radius / do I have an escape?") is not survivable: ε-exploration drops
a bomb, escaping is not learnable, and the agent dies. Measured on task 1: **100 % `KILLED_SELF`
over 1000 episodes**, gone the moment `BOMB` was masked out (`experiments/maxi.md` E01). On task 1
`BOMB`/`WAIT` are useless anyway (no crates); they come back on task 2 **together with** the
danger features, never before.

## Measurement (`tools/`) — read before running experiments
`evaluate.py` writes per-round, per-agent CSV + `.meta.json` (git commit, seed, snapshot of the
*rules* in `settings.py`). `analyze.py` gives means with 95 % bootstrap CI, `--compare A B`
(paired), `--ablation BASE V1 V2 …`, `--markdown`, `--plot`. `trainlog.py` writes one row per
episode; import it defensively in `train.py`
(`try: from tools.trainlog import TrainLogger / except ImportError: …`), since `tools/` is not
submitted. Plotting needs matplotlib (`uv add matplotlib`) — keep it out of `requirements.txt`.

```bash
uv run python tools/evaluate.py --agents <ours> --opponents rule_based --n-rounds 300 --label <name>
uv run python tools/analyze.py results/eval/<name>.csv
uv run python tools/analyze.py --compare results/eval/<old>.csv results/eval/<new>.csv --markdown
uv run python tools/analyze.py --ablation results/eval/<base>.csv results/eval/<v1>.csv \
    results/eval/<v2>.csv --metric score --plot
uv run python tools/trainlog.py results/train/*.csv --metric score
```
`--opponents` presets: `none`, `random`, `peaceful`, `coin_collector`, `mixed`, `rule_based`
(mapped to the task ladder). Non-default output needs `--out-dir`, e.g. `results/eval/task2_crates`.

Conventions we all follow, otherwise the numbers are not comparable:
- **Never change `--seed`.** Default `20260731`. `evaluate.py` reseeds per round
  (`base_seed + round_index`) so every agent ever measured sees identical arenas — that is what
  makes `--compare` paired. `main.py --seed` alone does **not** achieve this.
- **300 rounds** for any reported number, 100 for a quick check, 1000 for the final measurement.
  Naming: `results/eval/<person>_<model>_<version>__<task>.csv`.
- **A change counts as an improvement only if the paired 95 % CI excludes 0.** Otherwise it is
  "not demonstrated". Negative results stay in the report.
- **A training curve is not a result.** Nothing is claimed until measured at ε = 0 with
  `evaluate.py` — a broken policy looks healthy in the training log.
- **`steps` is only meaningful over completed rounds**, and from task 2 on completion rate is
  only meaningful next to `survived` — a dead agent also ends the round, so taken alone
  completion rate ranks the worst agent in the field first.
- **`analyze.py` assumes higher is better**, so it prints `WORSE` for a *falling* `steps`.
  Read that row inverted wherever efficiency is the goal.
- **Rung-3+ arms must be compared across several training seeds, never on a single run** —
  `main.py` does not seed the provided opponents, so two runs of the same command give
  different tables. Evaluations are **only partly** reproducible: `evaluate.py` seeds the
  opponents' `np.random` per round, but `coin_collector_agent` and `rule_based_agent` also
  shuffle with the **stdlib** `random`, which that does not touch — 22.7 % of rounds repeat
  exactly, the means repeat to four decimals. Rung-3 pairing is on arenas only
  (`MEASUREMENT.md`).
- Primary metric `score`. Per-task sets via `analyze.py --preset task1…task4`:
  task 1 `coins`/`steps`/`invalid` · task 2 `score`/**`suicides`**/`crates`/`bombs`/`survived` ·
  task 3 `score`/`kills`/**`suicides`**/`survived` ·
  task 4 `score`/**`won`**/`kills`/`suicides`/`killed_by`/`think_ms`.
- `suicides` changes role from task 3 on: progress signal on task 2 (must fall), regression guard
  afterwards (learning aggression is exactly when an agent forgets to run from its own bomb).
  On task 4 split the deaths: `suicides` (own bomb → escape logic) vs `killed_by` (opponent's
  bomb → positioning). Different bugs, different fixes.
- `won`/`rank` = standing within the round; on task 4 that matters more than mean score.
- Always watch `think_max_ms` (0.5 s tournament limit).

## Hints that matter for the grade
- The **scientific method is the main grading criterion**: subgoals, controlled experiments,
  defined metrics, each change justified by the previous round of testing. "Experiments and
  Results" is the most important report section.
- Feature engineering beats model complexity: situational awareness, pathfinding to the nearest
  coin, life-saving features (in blast radius / can I escape).
- Reward shaping: dense, balanced (penalize the opposite of every rewarded action), many custom
  events are fine. Potential-based shaping (Ng et al. 1999) depends on *states*, not actions.
  Auxiliary rewards do **not** exist in official games — don't overfit to them.
- Exploit the board's rotational/mirror symmetries for augmentation or state canonicalization.
- Hyperparameter optimization is explicitly graded — budget time for it.
- Deep learning is allowed but has historically failed to converge before the deadline.
- Test the submission in the provided `Dockerfile` (`docker build .`); list extra libraries in
  `requirements.txt` + README + report.

## Repo conventions
- Python 3.12, `uv` (`pyproject.toml`, `uv.lock`); add deps with `uv add`.
- Our agents live in `agent_code/` alongside the provided ones. `tpl_agent` is the template;
  `rule_based_agent` is the strong reference opponent and good training data — but our submitted
  agent must be *learned*, not rule-based.
- Naming: per-person development folders are `<person>_task<task>` (`benedict_task2`); the
  agreed, merged baseline for a task is `<method>_task<task>` (`tabular_q_task1`, later
  `dqn_task2`) and is **frozen** once agreed. Each task gets its own folder so two tasks can be
  compared without checking out an old commit.
- **`agent_code/tabular_q_task1/` is the agreed task-1 baseline** and the starting point for
  everything after it — 50.00 coins in every round at 123.7 steps, faster than
  `coin_collector_agent`'s 125.3. What it merges and why: `experiments/task1.md`.
  `benedict_coin_collector` and `maxi_coin_collector` stay as frozen evidence for their
  ledgers — do not develop in them.
- `tools/` holds our measurement chain; it is **not** submitted, so nothing in
  `agent_code/<name>/callbacks.py` may import from it.
- `results/eval/` CSVs and `results/train/` logs are grouped per task with the same names in both
  trees (`task1_coin_collectors/`, `task2_crates/`, `task3_opponents/`), plus
  `results/eval/baselines/` for the provided agents — a provided agent measured *in an opponent
  field* is a reference, not a result (`ref_<agent>__task3_<field>.csv`). Underscores, never
  spaces: a directory with a space silently breaks unquoted globs in analysis scripts.
  What is committed and what is gitignored: `MEASUREMENT.md`.
- `experiments/<person>.md` is the hand-written results ledger: one entry per experiment with
  question, change, **prediction written before the run**, commit hash, numbers and verdict. The
  CSVs are raw data; this is the narrative the report's Experiments chapter is built from, and
  nothing in `tools/` can reconstruct it after the fact. Same ownership rule as the logbooks —
  only append to your own.
- Restore original `settings.py` values before submitting if changed for training. Better still,
  make a training-only change an *environment switch* whose default is the upstream value
  (`BM_QUIET_LOGS`), so there is nothing to remember — `.meta.json` snapshots the rules, not the
  whole file, so an edited log level is invisible in the metadata.

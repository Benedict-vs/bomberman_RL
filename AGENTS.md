# Bomberman RL — Final Project (MLE, Uni Heidelberg, SS 2026)

Train an RL agent to play a 4-player Bomberman variant. Tournament + report.
Source of truth: `final_project.pdf`.
Approach, reasoning and milestones: `KONZEPT.md` (this file is the terse spec).

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

## Game rules (from `settings.py`)
- Board 17×17, `MAX_STEPS = 400`, up to 4 agents starting in corners.
- `BOMB_TIMER = 4` steps; explosion reaches `BOMB_POWER = 3` tiles in 4 directions, blocked by
  stone walls, does **not** turn corners; lingers `EXPLOSION_TIMER = 2`.
- Rewards: coin `+1`, kill `+5`. Killing opponents decides the score.
- Agent step time limit: `TIMEOUT = 0.5 s` (no limit while training). Exceeding it → `WAIT`, and
  the overrun is subtracted from the next step's budget.
- Tournament scenario: `classic` (`CRATE_DENSITY = 0.75`, `COIN_COUNT = 9`).
  Dev scenarios: `empty`, `coin-heaven`, `loot-crate`.
- Reference hardware: one thread of an AMD Ryzen 5 2600, ≤8 GB RAM.

## Agent interface — `agent_code/<name>/`
`callbacks.py` (always loaded):
```python
def setup(self): ...              # once, before first round; self persists across callbacks
def act(self, game_state: dict) -> str  # 'UP','DOWN','LEFT','RIGHT','BOMB','WAIT'
```
`self.logger` (logging.Logger) and `self.train` (bool) are preset.

`train.py` (loaded only with `--train N`):
```python
def setup_training(self): ...
def game_events_occurred(self, old_game_state, self_action, new_game_state, events): ...
def end_of_round(self, last_game_state, last_action, events): ...
```

`game_state` keys: `round`, `step`, `field` (`np.array(w,h)`: 1 crate, −1 wall, 0 free,
**image coords (x,y)** → printed transposed vs. GUI), `bombs` `[((x,y), t)]` (t==0 → about to
explode), `explosion_map` (steps of explosion remaining per tile), `coins` `[(x,y)]`,
`self` `(name, score, bomb_possible, (x,y))`, `others` `[same]`, `user_input`.

Events (`import events as e`): MOVED_{LEFT,RIGHT,UP,DOWN}, WAITED, INVALID_ACTION, BOMB_DROPPED,
BOMB_EXPLODED, CRATE_DESTROYED, COIN_FOUND, COIN_COLLECTED, KILLED_OPPONENT, KILLED_SELF,
GOT_KILLED, OPPONENT_ELIMINATED, SURVIVED_ROUND.

## Commands
```bash
python main.py play                                     # watch rule_based_agent
python main.py play --my-agent my_agent                 # vs 3 rule_based_agents
python main.py play --agents my_agent random_agent rule_based_agent --train 1
python main.py play --no-gui --agents my_agent --train 1 --scenario coin-heaven
python main.py replay <stored-replay>                   # needs --save-replay when playing
python main.py play --help
```
Useful flags: `--no-gui` (fast training), `--skip-frames`, `--seed` (fixes crates/coins, not agent
RNG), `--turn-based` (with `user_agent`), `--continue-without-training`.
Logs land in `agent_code/<name>/logs/<name>.log`; levels in `settings.py`.

## Models we are building
- **Model A — tabular Q-learning on hand-built features.** Lecture technique, fast to converge,
  interpretable. The baseline everything is measured against and the tournament fallback.
- **Model B — DQN on the raw board** (7 channels × 17×17, small CNN). We have GPU access for
  training; inference in the tournament is CPU-only. Historically the risky option — clear
  go/no-go: if it does not beat `rule_based_agent` in time, Model A is submitted.
  Model B still goes into the report either way; a documented failure is a valid result.
- Feature extraction, reward scheme, evaluation and training logs are **shared** between both.
  Splitting the team per model is explicitly forbidden by the task description.

## Measurement (`tools/`) — read before running experiments
Details and rationale in `KONZEPT.md` §6. The short version:

- **`tools/evaluate.py`** — per-round, per-agent statistics as CSV + `.meta.json`
  (git commit, seed, a snapshot of `settings.py`). Drives `BombeRLeWorld` directly and
  touches no framework file, so it survives the tournament reset.
  `main.py --save-stats` is *not* enough: it only writes lifetime totals per agent and
  per-round totals summed over all agents, so no per-agent confidence interval is possible.
- **`tools/analyze.py`** — means with 95 % bootstrap CI; `--compare A B` does a **paired**
  comparison; `--ablation BASE V1 V2 …` compares many variants against one baseline on a
  single metric; `--markdown` emits report-ready tables; `--plot` writes figures to
  `results/figures/` (bar chart with CIs for a summary, forest plot for comparison and
  ablation). Plotting needs matplotlib: `uv add matplotlib` (analysis only — keep it out of
  the submitted `requirements.txt`).
- **`tools/trainlog.py`** — one row per episode for learning curves; import defensively in
  `train.py` (`try: from tools.trainlog import TrainLogger / except ImportError: ...`),
  since `tools/` is not part of the submission.

Conventions we all follow, otherwise the numbers are not comparable:
- **Never change `--seed`.** Default `20260731`. The harness reseeds the world RNG per round
  (`base_seed + round_index`), which makes every run use *identical* arenas — that is what
  makes comparisons paired and the CIs tight. `main.py --seed` alone does **not** achieve this:
  the world draws from the same RNG every step, so arenas drift apart from round 2 onwards.
- **300 rounds** for any number that gets reported, 100 for a quick check, 1000 for the final
  measurement. Naming: `results/eval/<person>_<model>_<version>__<task>.csv`.
- **A change counts as an improvement only if the paired 95 % CI excludes 0.** Otherwise it is
  "not demonstrated". Negative results stay in the report.
- **A training curve is not a result.** Nothing is claimed until it has been measured with
  `tools/evaluate.py` at ε = 0. While training, ε-exploration and the still-changing table keep
  breaking the policy out of cycles, so a broken agent can look healthy in the log. Measured
  2026-08-05: 48.2 coins/episode at the end of training, **1.45** in the evaluation of the same
  model (`experiments/maxi.md` E01).
- **`steps` only measures path length in rounds that were completed.** The round ends the moment
  the last coin is collected, so 400 means "never finished", not "slow", and a mean over both
  kinds of round is meaningless. Always report the **completion rate** next to it, or restrict
  `steps` to completed rounds (`tools/plot_task1_versions.py` does the latter). This is what
  distinguishes "navigates badly" from "navigates fine but gets stuck".
- **`analyze.py` assumes higher is better**, so it prints `WORSE` for a *falling* `steps`.
  On task 1 (and anywhere else efficiency is the goal) read that row inverted.
- Primary metric `score`. Per-rung metric sets via `analyze.py --preset task1…task4`:
  task 1 `coins`/`steps`/`invalid` · task 2 `score`/**`suicides`**/`crates`/`bombs`/`survived` ·
  task 3 `score`/`kills`/**`suicides`**/`survived` ·
  task 4 `score`/**`won`**/`kills`/`suicides`/`killed_by`/`think_ms`.
- `suicides` changes role from task 3 on: progress signal on task 2 (must fall), regression
  guard afterwards (must not creep back up — learning aggression is exactly when an agent
  forgets to run from its own bomb).
- On task 4 split the deaths: `suicides` (own bomb → escape logic broken) vs `killed_by`
  (opponent's bomb → positioning/danger awareness). Different bugs, different fixes.
- `won`/`rank` = standing within the round. On task 4 this matters more than mean score —
  the tournament is decided against the other agents.
- Always watch `think_max_ms` (0.5 s tournament limit).

```bash
uv run python tools/evaluate.py --agents <ours> --opponents rule_based --n-rounds 300 --label <name>
uv run python tools/analyze.py results/eval/<name>.csv
uv run python tools/analyze.py --compare results/eval/<old>.csv results/eval/<new>.csv --markdown
uv run python tools/analyze.py --ablation results/eval/<base>.csv results/eval/<v1>.csv \
    results/eval/<v2>.csv --metric score --plot
uv run python tools/trainlog.py results/train/*.csv --metric score
```

Opponent presets for `--opponents`: `none`, `random`, `peaceful`, `coin_collector`, `mixed`,
`rule_based` (mapped to the task ladder below).

**Measurements are versioned** (settled 2026-08-04, was the open item in `KONZEPT.md` §6.7).
`.gitignore` now excludes only `results/figures/`; `results/eval/` and `results/train/` are
committed. They are small text files and they are the evidence for the report's most important
chapter. Training logs especially: `--seed` fixes the arenas but *not* the agent's exploration
RNG, so a lost learning curve cannot be reproduced, only replaced by a different one. Figures
stay ignored — they are pure functions of the CSVs.

## Task ladder (subsets of each other)
1. `coin-heaven`, no crates/opponents → efficient navigation to revealed coins.
2. `classic`, no opponents → use bombs to open crates, **escape own bombs**, keep navigating.
3. Crates + `peaceful_agent` (easy) / `coin_collector_agent` (hard) → hunt and kill.
4. Crates + `rule_based_agent` → must beat it to have a shot at the tournament.

**Keep the action set in step with the features.** `BOMB` in the action space without a danger
feature ("am I in a blast radius / do I have an escape?") is not survivable: ε-exploration drops
a bomb, escaping is not learnable, and the agent dies. Measured on rung 1: **100 % `KILLED_SELF`
over 1000 episodes**, gone the moment `BOMB` was masked out (`experiments/maxi.md` E01). On rung 1
`BOMB`/`WAIT` are useless anyway (no crates); they come back on rung 2 **together with** the
danger features, never before.

## Hints that matter for the grade
- The **scientific method is the main grading criterion**: subgoals, controlled experiments,
  defined performance metrics, each change justified by the previous round of testing.
  "Experiments and Results" is the most important report section.
- Feature engineering beats model complexity: situational awareness (walls adjacent), pathfinding
  (direction to nearest coin), life-saving (in blast radius / can I escape).
- Reward shaping: dense, balanced (penalize the opposite of every rewarded action), many custom
  events are fine. Potential-based shaping (Ng et al. 1999) depends on *states*, not actions.
  Auxiliary rewards do **not** exist in official games — don't overfit to them.
- Exploit the board's rotational/mirror symmetries for data augmentation or state canonicalization.
- Hyperparameter optimization is explicitly graded — budget time for it.
- Deep learning is allowed but has historically failed to converge before the deadline; simple,
  well-tuned models have won.
- Test the submission in the provided `Dockerfile` (`docker build .`), and list extra libraries in
  `requirements.txt` + README + report.

## Repo conventions
- Python 3.12, `uv` (`pyproject.toml`, `uv.lock`); add deps with `uv add`.
- Our agents live in `agent_code/` alongside the provided ones (`tpl_agent` is the template;
  `rule_based_agent` is the strong reference opponent and a good source of training data —
  but our submitted agent must be *learned*, not rule-based).
- **`agent_code/tabular_q_task1/` is the agreed rung-1 baseline** and the starting point for
  everything after it. It merges the two independently developed coin collectors on the
  evidence of both ledgers (BFS coin direction and random tie-breaking from Maxi's; per-cell
  learning rate `1/N(s,a)^0.7`, the full six-action set, the mixed-radix table and the seeded
  measurement harness from Benedict's). 50.00 coins in every round at 123.7 steps — **faster
  than `coin_collector_agent`'s 125.3**, which is the metric that discriminates on this rung.
  Rationale, findings and reproduction: `experiments/task1.md`.
  Each rung gets its own folder so two rungs can be measured against each other without
  checking out an old commit. Per-person development folders are `<person>_task<rung>`
  (`benedict_task2`); the agreed, merged baseline for a rung is `<method>_task<rung>`
  (`tabular_q_task1`, later `dqn_task2`) and is frozen once it is agreed.
  `benedict_coin_collector` and `maxi_coin_collector` stay as frozen evidence for their
  ledgers — do not develop in them.
- `tools/` holds our measurement chain; it is **not** submitted, so nothing in
  `agent_code/<name>/callbacks.py` may import from it.
- `results/eval/` evaluation CSVs, `results/train/` training logs. Both written by `tools/`.
  Grouped per rung, same names in both trees: `task1_coin_collectors/`, `task2_crates/`,
  plus `results/eval/baselines/` for the provided agents. Underscores, never spaces — a
  directory with a space in it silently breaks unquoted globs in analysis scripts.
  Non-default locations need `--out-dir`, e.g.
  `--out-dir results/eval/task2_crates`.
- `experiments/<person>.md` is the hand-written results ledger: one entry per experiment with
  question, change, **prediction written before the run**, commit hash, numbers and verdict.
  The CSVs are raw data; this is the narrative the report's Experiments chapter is built from,
  and nothing in `tools/` can reconstruct it after the fact. Same ownership rule as the
  logbooks — only append to your own.
- Restore original `settings.py` values before submitting if changed for training.
  `tools/evaluate.py` records the active settings in its `.meta.json` so a mismatch is visible.

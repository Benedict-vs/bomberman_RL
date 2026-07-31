# Bomberman RL — Final Project (MLE, Uni Heidelberg, SS 2026)

Train an RL agent to play a 4-player Bomberman variant. Tournament + report.
Source of truth: `final_project.pdf`.

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

## Task ladder (subsets of each other)
1. `coin-heaven`, no crates/opponents → efficient navigation to revealed coins.
2. `classic`, no opponents → use bombs to open crates, **escape own bombs**, keep navigating.
3. Crates + `peaceful_agent` (easy) / `coin_collector_agent` (hard) → hunt and kill.
4. Crates + `rule_based_agent` → must beat it to have a shot at the tournament.

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
- Restore original `settings.py` values before submitting if changed for training.

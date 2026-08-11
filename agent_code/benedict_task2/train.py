"""Q-learning updates for `benedict_task2`.

Loaded only with `--train`, so nothing here runs in the tournament. That is why
`tools/` may be imported (defensively) and why the exploration RNG lives here.

Reproducing the shipped `q_table.npy`
-------------------------------------
Every default below is the shipped configuration, so this one command rebuilds
it bit for bit (the suffix only keeps it from overwriting the shipped file,
which `save_table` refuses to do anyway)::

    BM_MODEL_SUFFIX=_repro uv run python main.py play --no-gui \\
        --agents benedict_task2 --train 1 --n-rounds 20000 --seed 810731

    cmp checkpoints/benedict_task2/q_table_repro.npy \\
        agent_code/benedict_task2/q_table.npy

Two things that are *not* obvious and are easy to get wrong:

- **The world seed 810731 is not optional and is not the evaluation seed.**
  Reproduction needs both it and `TRAIN_SEED + RUN_INDEX` below; either alone
  leaves runs incomparable. Never train on 20260731 -- the agent would then be
  measured on arenas it trained on.
- **Training is coarse-to-fine.** `WARM_SUFFIX` starts the 64 000-row table from
  a converged 12 800-row parent (E23). That parent belongs to the *pre-E20*
  feature map and cannot be retrained at this commit, which is why it is the one
  checkpoint kept under version control.

Why these hyperparameters (evidence in `experiments/benedict_task2.md` §4)
--------------------------------------------------------------------------
=================  ==========  ====================================================
setting            value       why
=================  ==========  ====================================================
alpha              1/N^0.7     Rung 1's largest single effect: constant alpha gave
                               20.78 +- 6.98 coins, per-cell 49.15 +- 1.23. A
                               constant alpha meets neither of L26's convergence
                               conditions, and an unsettled cell here is not a small
                               error but an absorbing deadlock. Exponent swept in
                               E22: 1.0 tripled the thin-margin fraction and cost
                               the best seed 88 crates -- the target is
                               non-stationary, so sample-averaging is wrong.
gamma              0.99        E15. At 0.9 the horizon is ~10 steps, shorter than
                               the distance to most targets; 0.99 cut the crate std
                               from 24.2 to 2.9 and removed a peak-then-decay that
                               three earlier entries blamed on their own changes.
eps                0.2 -> 0.02 E15/E18. A floor of 0.10 halves performance; 0.005 is
                               indistinguishable from 0.02; 0 stops learning outright
                               (4 of 5 tables frozen from 40 000 on). The residual
                               0.64 deaths/episode are the tuition that keeps rare
                               rows alive.
COIN_COLLECTED     +5          E16, and the most surprising result on the rung: the
                               game's own +1 costs 45 crates. The reward is not a
                               statement about coins being valuable, it is what keeps
                               the value function separated -- 16.7:1 against the
                               crate reward gives the table the dynamic range that
                               stops near-ties being settled by noise.
CRATE_DESTROYED    +0.3        E16. 1.0 halves the crate count -- not by killing the
                               agent (98 % of the deficit is in rounds nobody died)
                               but by degrading placement: it bombs 24 % more often
                               for 1.09 crates a bomb instead of 2.55.
KILLED_SELF        -5          Rung 1. Removed every suicide at once, which is why
                               BOMB can stay in the action set instead of being
                               masked out. E17 confirms it is load-bearing.
STEP_COST          -0.1        Shortest-path pressure; the round is capped at 400
                               steps and 99 % of rounds hit that cap.
WARM_N             100         E23. Not optional: alpha is exactly 1 on a cell's
                               first update, so an untouched transfer is overwritten
                               immediately. 10 000 and 100 000 are both worse -- the
                               table then cannot differentiate the rows the new digit
                               created.
episodes           20 000      E23. Longer is worse on this map: 40 000 loses ~7
                               crates, and E18 measured 200 000 turning the coarse
                               map from 97.31 into 62.85 by re-rolling near ties.
=================  ==========  ====================================================
"""

import atexit
import json
import os
from typing import List

import numpy as np

import events as e
from .callbacks import (state_to_features, ACTIONS, MODEL_FILE, N_STATES,
                        DELTAS, FEATURE_SIZES)

try:
    from tools.trainlog import TrainLogger
except ImportError:     # tools/ is not part of the submission
    TrainLogger = None


AGENT_NAME = "benedict_task2"

STEP_COST = -0.1    # encourages shorter paths
# 0.99, not the 0.9 carried since E01: E15 measured the crate std falling from
# 24.2 to 2.9 and the peak-then-decay of E12/E13/E14 disappearing. At gamma=0.9
# the horizon is ~10 steps, shorter than the distance to most BFS targets.
GAMMA = float(os.environ.get("BM_GAMMA", 0.99))

ALPHA = 0.1         # only used when ALPHA_MODE == "const"
ALPHA_EXP = float(os.environ.get("BM_ALPHA_EXP", 0.7))     # in (0.5, 1]

EPS_START = 0.2
EPS_END = float(os.environ.get("BM_EPS_END", 0.02))

# Rounds between model saves. The table is 2.9 MB since E20, so saving every
# round is ~120 GB of writes over a 40 000-round run, with up to fifteen of them
# running at once -- enough I/O to dominate the batch. A crash costs at most this
# many rounds, and the atexit hook in setup_training covers every normal ending.
SAVE_EVERY = 100

# Episodes at which the table is also written to its own file, so the learning
# curve can be measured at eps = 0 afterwards instead of read off the training
# log. Learning is untouched -- these are extra writes, not extra updates, so
# every checkpoint is exactly the table a run of that length would have left.
CHECKPOINTS = (20_000, 40_000, 70_000, 100_000, 150_000, 200_000, 300_000)

# The two swept in E16. Both were guesses -- the coin in E01, the crate in E10 --
# and E15's change of gamma rescaled every reward against the step cost by a
# factor of ten, so the balance they struck at gamma=0.9 no longer holds.
REWARDS = {
    e.COIN_COLLECTED: float(os.environ.get("BM_COIN", 5)),
    e.CRATE_DESTROYED: float(os.environ.get("BM_CRATE", 0.3)),
    e.INVALID_ACTION: -1,
    e.WAITED: -0.1,
    e.KILLED_SELF: -5,
}

# --- Experiment switches --------------------------------------------------
# Read from the environment so a shell loop can sweep seeds and arms without
# editing this file. Every default is the shipped configuration.

# Seeds the exploration RNG as TRAIN_SEED + RUN_INDEX. 5 is the seed the shipped
# table came from, selected on the held-out world seed 550731 (E23); 0-4 are the
# sweep it was selected against.
RUN_INDEX = int(os.environ.get("BM_RUN_INDEX", 5))

ALPHA_MODE = os.environ.get("BM_ALPHA", "visit")    # "visit" | "const" (E06)
EPS_MODE = os.environ.get("BM_EPS", "decay")        # "decay" | "const" (E05)

# E19, rejected: potential-based shaping, F = gamma*Phi(s') - Phi(s) with
# Phi = -BM_SHAPE * (BFS distance to digit 6's goal). Kept at its default of 0 --
# which skips the whole mechanism, BFS included -- so E19 stays reproducible.
# It fails because a row of this table is a bucket of states with different Phi,
# so the offset does not cancel between actions the way the theorem needs.
SHAPE = float(os.environ.get("BM_SHAPE", 0))

# E23: coarse-to-fine value transfer. Names a table the way BM_MODEL_SUFFIX does
# and every row starts from the row it was split from. Empty starts from zero,
# which is ~20 crates worse -- see the module docstring.
#
# CAREFUL when reproducing an entry older than E23: those runs had no warm start,
# and the launch commands recorded in the ledger do not set this variable, so at
# this commit they would silently get one. Pass BM_WARM= (empty) for E08-E22.
WARM_SUFFIX = os.environ.get("BM_WARM", "_e16_c5_k03_s0__ep100000")
WARM_N = int(os.environ.get("BM_WARM_N", 100))

EPS_DECAY = 0.9995 if EPS_MODE == "decay" else 1.0
TRAIN_SEED = 20260731

# Change per experiment. The training log is *appended* to, so a stale value here
# silently merges two runs into one file (cost half an hour to unpick in E05b).
EXPERIMENT = "e23"
ARM = os.environ.get("BM_ARM", "")
RUN_NAME = f"q_{EXPERIMENT}{'_' + ARM if ARM else ''}_s{RUN_INDEX}"


def setup_training(self):
    """Called once before the first round, after `setup` in callbacks.py."""

    self.rng = np.random.default_rng(TRAIN_SEED + RUN_INDEX)
    self.eps = EPS_START
    self.gamma = GAMMA
    self.visits = np.zeros((N_STATES, len(ACTIONS)), dtype=np.int64)
    if WARM_SUFFIX:
        warm_start(self)
    self.last_phi = None
    
    # E20 post-mortem: a diagnostic that drives BombeRLeWorld with train=True
    # while BM_MODEL_SUFFIX names a real checkpoint will overwrite it, because
    # MODEL_FILE resolves to that path and the atexit hook below fires on any
    # normal exit. It happened to q_table_e16_c5_k03_s2__ep100000.npy, and it
    # was caught by an mtime rather than by anything in this file. Remember
    # whether the table existed before this run started; save_table refuses if
    # it did.
    self.model_file_preexisted = os.path.isfile(MODEL_FILE)

    # Per-episode accumulators for the learning curve. evaluate.py measures the
    # finished agent; this is what shows whether it converged, and when.
    self.trainlog = TrainLogger(
        agent=AGENT_NAME,
        run=RUN_NAME,
        # Rung-2 logs go beside the rung-2 evaluations; see AGENTS.md.
        # TrainLogger anchors a relative out_dir to the repo root -- agents.py
        # chdirs into this folder around every callback, so the cwd is not it.
        out_dir="results/train/task2_crates",
        hyperparams={"alpha": ALPHA, "alpha_mode": ALPHA_MODE, "alpha_exp": ALPHA_EXP,
                     "eps_start": EPS_START, "eps_mode": EPS_MODE, "eps_end": EPS_END,
                     "eps_decay": EPS_DECAY,
                     "gamma": GAMMA, "shape": SHAPE,
                     "warm": WARM_SUFFIX, "warm_n": WARM_N,
                     "train_seed": TRAIN_SEED + RUN_INDEX, "run_index": RUN_INDEX,
                     "step_cost": STEP_COST,
                     "rewards": {k: v for k, v in REWARDS.items()},
                     "n_states": len(self.q),
                     "features": "4 neighbour states (blocked/lethal/in-blast/clear) "
                                 "+ own grace period + BFS direction to coin-or-crate "
                                 "+ bomb-here-pays-off"},
        extra_columns=["td_error"],
    ) if TrainLogger else None
    self.episode_events = []
    self.episode_reward = 0.0
    self.episode_td = []

    # environment.py:112 runs the agent in-process (SequentialAgentBackend), so
    # this fires on any normal exit. Without it, a round count that is not a
    # multiple of SAVE_EVERY would silently ship a table up to SAVE_EVERY
    # rounds stale -- the kind of mismatch that is invisible until a run does
    # not reproduce.
    atexit.register(save_table, self)


def save_table(self) -> None:
    """Write the table, unless that would clobber a checkpoint we did not create.

    A training run legitimately overwrites its own file every SAVE_EVERY rounds
    -- the guard is armed once, at setup, and disarmed by the first successful
    write, so only the *first* write of a run can trip it. That is the write
    that destroys somebody else's result.

    Deliberately a hard failure. The alternative -- warn and continue -- is what
    the E18 missing-table fallback did, and it produced ten evaluations of an
    all-zero table before anyone noticed.
    """

    if self.model_file_preexisted:
        raise FileExistsError(
            f"{MODEL_FILE} existed before this run started. Training would "
            "overwrite it. Set BM_MODEL_SUFFIX to a name of this run's own, or "
            "delete the file deliberately if the overwrite is intended."
        )
    # checkpoints/<agent>/ is where BM_MODEL_SUFFIX now points; it is gitignored
    # and may not exist on a fresh clone.
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    np.save(MODEL_FILE, self.q)
    write_layout(MODEL_FILE)
    self.model_file_preexisted = False


def checkpoint_file(episode: int) -> str:
    """`q_table<suffix>__ep<N>.npy` -- exactly the path that
    `BM_MODEL_SUFFIX=<suffix>__ep<N>` resolves to, so a checkpoint is evaluated
    by setting that one variable and `callbacks.py` needs no special case.
    """

    base, ext = os.path.splitext(MODEL_FILE)
    return f"{base}__ep{episode}{ext}"


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """One Q-learning update per step."""

    reward = reward_from_events(self, events)

    # Tally before the guard: the first call of a round has no old state to
    # learn from, but its events still happened and belong in the episode total.
    self.episode_events.extend(events)
    self.episode_reward += reward

    if old_game_state is None:
        return

    s = state_to_features(old_game_state)
    s_next = state_to_features(new_game_state)
    a = ACTIONS.index(self_action)

    # E19: shaping enters the update only, never the logged reward. The cache saves
    # one BFS per step: last step's new state is this step's old state.
    shaping = 0.0
    if SHAPE:
        phi_old = self.last_phi if self.last_phi is not None else phi(old_game_state)
        phi_new = phi(new_game_state)
        self.last_phi = phi_new
        shaping = self.gamma * phi_new - phi_old

    td_target = reward + shaping + self.gamma * np.max(self.q[s_next])

    td_error = td_target - self.q[s, a]
    self.q[s, a] += learning_rate(self, s, a) * td_error

    self.episode_td.append(abs(td_error))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Terminal update, model save, and one row in the training log."""

    s = state_to_features(last_game_state)
    a = ACTIONS.index(last_action)
    reward = reward_from_events(self, events)

    # E19: transition into terminal -- Phi(terminal) = 0, so F = -Phi(s_last).
    # Recomputed, never taken from the cache. The cache is correct exactly when
    # the agent *died*: `send_game_events` (environment.py:468) skips a dead
    # agent, so no step update fired and it still holds Phi of the state this
    # update touches. On a *surviving* round the step update did fire and left
    # Phi of the post-step state behind, while the cell updated here is the
    # pre-step one. Probed over 40 rounds with a trained table: exact on 23/23
    # deaths, wrong on 16 of 17 survivals, by up to 0.4 at SHAPE = 0.2 -- and
    # survival is the common case (>= 0.987), so this was the usual path.
    shaping = -phi(last_game_state) if SHAPE else 0.0
    self.last_phi = None

    td_target = reward + shaping

    td_error = td_target - self.q[s, a]
    self.q[s, a] += learning_rate(self, s, a) * td_error

    round_no = last_game_state["round"]
    if round_no % SAVE_EVERY == 0:
        save_table(self)
    if round_no in CHECKPOINTS:
        np.save(checkpoint_file(round_no), self.q)

    self.episode_events.extend(events)
    self.episode_reward += reward
    self.episode_td.append(abs(td_error))
    if self.trainlog:
        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=last_game_state["self"][1],
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.eps,
            extra={"td_error": float(np.mean(self.episode_td))},
        )
    self.episode_events = []
    self.episode_reward = 0.0
    self.episode_td = []

    # After logging, so the recorded epsilon is the one that generated the episode.
    self.eps = max(EPS_END, self.eps * EPS_DECAY)


LAYOUT_EXT = ".layout.json"


def write_layout(table_file: str) -> None:
    """Record the feature layout beside a saved table.

    A `.npy` says how many rows it has and nothing about what they mean. Two
    different `FEATURE_SIZES` can produce the same row count, so a table alone
    cannot tell a warm start whether it is a legitimate parent -- see the
    append-only argument in `warm_start`. One line of JSON removes the guesswork.
    """

    with open(table_file + LAYOUT_EXT, "w") as fh:
        json.dump({"feature_sizes": list(FEATURE_SIZES)}, fh)


def read_layout(table_file: str) -> list | None:
    """The layout a table was trained on, or None for tables written before this
    existed -- including the E16 parent the shipped model starts from."""

    try:
        with open(table_file + LAYOUT_EXT) as fh:
            return json.load(fh)["feature_sizes"]
    except (OSError, KeyError, ValueError):
        return None


def warm_start(self) -> None:
    """Initialise this table from a coarser one, each row from its parent.

    E20 measured the deficit this addresses: the distance digit splits every old
    row into five, and on two seeds of five the run filled only ~485 of its ~840
    used rows -- a 64 000-row map carrying the information of a 12 800-row one.
    The coarse table already knows what those states are worth up to the
    distinction the new digit draws, so a child starts from its parent rather
    than from zero.

    **The new digits must be appended, never inserted.** `encode` is mixed radix,
    so appending digits to `FEATURE_SIZES` multiplies every old index by the new
    radices: parent row i becomes children `k*i .. k*i+k-1`, which is exactly
    `np.repeat`. Insert a digit anywhere else and the row count is identical, the
    divisibility check below still passes, and the mapping is silently wrong --
    measured on 5 000 random states, an inserted digit mis-maps 39 % of them.
    That is why the layout is written beside every table and checked here.

    Only cells whose parent carried value get the pseudo-count: crediting the
    rest would start genuinely new rows at a twenty-fifth of their learning rate
    for nothing.
    """

    coarse_file = os.path.join(os.path.dirname(MODEL_FILE),
                               f"q_table{WARM_SUFFIX}.npy")
    coarse = np.load(coarse_file)
    factor, remainder = divmod(N_STATES, len(coarse))
    if remainder:
        raise ValueError(
            f"{coarse_file} has {len(coarse)} rows, which does not divide the "
            f"current {N_STATES} -- it is not a parent of this feature map.")

    parent_sizes = read_layout(coarse_file)
    if parent_sizes is None:
        self.logger.warning(
            f"{coarse_file} has no {LAYOUT_EXT} sidecar, so the append-only "
            "assumption cannot be checked -- verify it by hand before trusting "
            "the result.")
    elif tuple(FEATURE_SIZES[:len(parent_sizes)]) != tuple(parent_sizes):
        raise ValueError(
            f"{coarse_file} was trained on FEATURE_SIZES {tuple(parent_sizes)}, "
            f"which is not a prefix of the current {tuple(FEATURE_SIZES)}. New "
            "digits must be appended, not inserted -- otherwise np.repeat maps "
            "parent rows onto the wrong children and nothing will complain.")

    self.q[:] = np.repeat(coarse, factor, axis=0)
    self.visits[np.repeat(np.abs(coarse).sum(axis=1) > 0, factor)] = WARM_N
    self.logger.info(f"Warm start from {coarse_file}: {len(coarse)} rows -> "
                     f"{N_STATES}, {int((np.abs(coarse).sum(axis=1) > 0).sum())} "
                     f"parents with value, pseudo-count {WARM_N}")


def learning_rate(self, s: int, a: int) -> float:
    """Per-cell learning rate. See the module docstring for why this matters."""

    self.visits[s, a] += 1
    if ALPHA_MODE == "visit":
        return 1.0 / self.visits[s, a] ** ALPHA_EXP
    return ALPHA


def reward_from_events(self, events: List[str]) -> float:
    return sum(REWARDS.get(ev, 0.0) for ev in events) + STEP_COST

def _bfs_distance(x: int, y: int, field, is_goal) -> int | None:
    """Steps until `is_goal` fires. Goals are tested but not expanded, the same
    rule `callbacks.bfs_first_step` uses, so a crate is a valid destination.
    No bounds check: only free tiles are expanded and the arena is walled all
    round, so the search never reaches an index off the board. Only differences
    of Phi enter the update, so the unit (steps) needs no normalisation."""

    queue = [(x, y, 0)]
    visited = {(x, y)}
    head = 0
    while head < len(queue):
        cx, cy, d = queue[head]
        head += 1
        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if is_goal((nx, ny)):
                return d + 1
            if field[nx, ny] != 0:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, d + 1))
    return None


def phi(game_state: dict) -> float:
    """−SHAPE · distance to digit 6's goal. Deliberately the safe branch only --
    danger states keep the same Φ (any state function is a valid potential; the
    escape behaviour is priced by KILLED_SELF, not by shaping)."""
    x, y = game_state["self"][3]
    field = game_state["field"]
    coins = set(game_state["coins"])
    if (x, y) in coins:
        return 0.0
    d = None
    if coins:
        d = _bfs_distance(x, y, field, lambda p: p in coins)
    if d is None:
        d = _bfs_distance(x, y, field, lambda p: field[p] == 1)
    return -SHAPE * (d or 0)

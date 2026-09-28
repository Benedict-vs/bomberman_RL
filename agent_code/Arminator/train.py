"""Q-learning updates for `Arminator`, the task-4 agent.

Loaded only with `--train`, so nothing here runs in the tournament. This is why
`tools/` may be imported here (inside a try block) and why the exploration RNG
is defined here.

Reproducing the shipped table:

    uv run python main.py play --agents Arminator \\
        rule_based_agent rule_based_agent rule_based_agent \\
        --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731

Every default in this file is the shipped configuration, so the command needs
no environment variables. It writes `q_table_trained.npy` next to the agent
instead of overwriting `q_table.npy`, so a training run cannot destroy the
table used in the tournament. To use a retrained table, copy it over
`q_table.npy` by hand.

The command does not reproduce the table bit for bit. `main.py` does not seed
the provided agents, and each of them reseeds the global RNG from OS entropy in
its own `setup`, which runs after ours. Two runs of the same command therefore
give different tables. The shipped table is one seed of a fifteen-seed sweep,
chosen on a validation world seed and confirmed on a held-out one. Arms should
be compared across the seeds of a sweep, never on a single run.
`experiments/benedict_task4.md` has the full protocol.

For the same reason evaluations are only partly reproducible: `evaluate.py`
seeds the opponents' `np.random`, but `rule_based_agent` also shuffles with the
stdlib `random`, which is not seeded, and only ~21 % of rounds repeat exactly.
The means do not repeat either. The same table evaluated twice over 1000 rounds
differs by 0.121 in mean score, so a 1000-round evaluation has a +/-0.12 noise
floor on `score` from the opponents alone. Pairing is on arenas.

The training world seed 810731 is not an evaluation seed. Never train on
20260731, 550731 or 990731, otherwise the agent is evaluated on arenas it was
trained on.

Why these hyperparameters
-------------------------
=================  ==========  ====================================================
setting            value       why
=================  ==========  ====================================================
alpha              1/N^0.7     The largest single effect on this project: a
                               constant alpha gave 20.78 +- 6.98 coins against
                               49.15 +- 1.23 per-cell. Constant alpha satisfies
                               neither Robbins-Monro condition, and a cell that
                               does not settle here causes an absorbing deadlock
                               rather than a small error. Exponent 1.0 tripled the
                               thin-margin fraction: the target is
                               non-stationary, so sample-averaging does not fit.
gamma              0.99        At 0.9 the horizon is ~10 steps, shorter than the
                               distance to most BFS targets. 0.99 cut the crate
                               standard deviation from 24.2 to 2.9.
eps                0.2 -> 0.02 A floor of 0.10 halves performance, 0.005 is
                               indistinguishable from 0.02, and 0 stops learning.
                               The remaining exploration deaths are the price for
                               keeping rare rows updated.
COIN_COLLECTED     +5          With the game's own +1 the agent destroys 45 fewer
                               crates. The value is not meant as the true worth
                               of a coin; it keeps the action values far enough
                               apart that near ties are not decided by noise.
CRATE_DESTROYED    +1.0        Solo, 1.0 halved the crate count by degrading bomb
                               placement, and 0.3 was better there. With three
                               opponents it is the other way round: nine coins
                               shared four ways cut gross earnings ~4x while
                               steps alive fall only 1.85x, so at 0.3 an
                               action-independent step cost dominates the return
                               and the fixed point becomes action-independent
                               too. Worth +0.93 score and +11.5 crates, paired.
STEP_COST           0          The same trade-off from the other side. Solo, the
                               shortest-path pressure helps (77.3 against 40.1);
                               in an opponent field the same table earns 18.3
                               against 26.1 and the decision margin collapses.
                               Zero here, with the crate reward at 1.0.
KILLED_SELF         0          `environment.py:264` adds GOT_KILLED to every agent
GOT_KILLED         -5          killed by a blast, and `:251` adds KILLED_SELF in
                               addition when the bomb was its own. A table with
                               both therefore charges a suicide their sum, which
                               is not the symmetric pricing it appears to be.
                               Putting the whole penalty on GOT_KILLED charges a
                               death once.
KILLED_OPPONENT     0          Tested at 5 and at 25 (25 keeps the game's 5:1
                               kill:coin ratio at this table's scale). Neither
                               changed kills: digit 7 is one bit shared between
                               "a bomb here opens a crate" and "a bomb here
                               catches an opponent", and crates are far more
                               common than kills, so the reward cannot change
                               the decision. That would need a new feature.
WARM_N             100         alpha is 1 on a cell's first update, so without a
                               pseudo-count the transferred value is overwritten
                               immediately. 10 000 and 100 000 are both worse,
                               because the table then cannot separate the rows a
                               new digit created.
episodes           20 000      Chosen by measurement. Training to 300 000 costs
                               -0.770 score [-1.355, -0.185], with 0 of 5 seeds
                               improving, and the loss comes from bomb
                               placement: crates per bomb 1.182 -> 0.988. The
                               decline is monotone from 20 000 on, so this is
                               the best of the tested lengths and not a limit
                               set by the compute budget.
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


AGENT_NAME = "benedict_task4"

# E27: an action-independent per-step cost shifts the fixed point without changing
# any decision. Shipped at 0; the rung-3 derivation is in benedict_task3.md.
STEP_COST = float(os.environ.get("BM_STEP_COST", 0))
# 0.99 instead of the 0.9 used since E01: in E15 the crate std fell from 24.2 to
# 2.9 and the peak-then-decay of E12/E13/E14 disappeared. At gamma=0.9 the
# horizon is ~10 steps, shorter than the distance to most BFS targets.
GAMMA = float(os.environ.get("BM_GAMMA", 0.99))

ALPHA_EXP = float(os.environ.get("BM_ALPHA_EXP", 0.7))     # in (0.5, 1]

EPS_START = 0.2
EPS_END = float(os.environ.get("BM_EPS_END", 0.02))

# Rounds between model saves. Since E20 the table is 2.9 MB, so saving every
# round means ~120 GB of writes over a 40 000-round run, with up to fifteen runs
# in parallel. That much I/O would dominate the batch. A crash loses at most this
# many rounds, and the atexit hook in setup_training handles every normal exit.
SAVE_EVERY = 100

# Episodes at which the table is also saved to a separate file, so the learning
# curve can be measured at eps = 0 afterwards instead of read from the training
# log. These are extra writes, not extra updates, so learning is unaffected and
# each checkpoint is the table a run of that length would have produced.
# Can be overridden so a long run samples its own horizon: BM_CHECKPOINTS=20000,40000,...
CHECKPOINTS = tuple(int(c) for c in
                    os.environ.get("BM_CHECKPOINTS", "5000,10000,20000").split(","))

# The two rewards swept in E16. Both were first set by guess (the coin in E01,
# the crate in E10), and E15's change of gamma rescaled every reward against the
# step cost by a factor of ten, so their balance at gamma=0.9 no longer held.
REWARDS = {
    e.COIN_COLLECTED: float(os.environ.get("BM_COIN", 5)),
    e.CRATE_DESTROYED: float(os.environ.get("BM_CRATE", 1.0)),
    e.INVALID_ACTION: -1,
    e.WAITED: -0.1,
    # E26 bug fix. `environment.py:264` adds GOT_KILLED to every agent killed by
    # a blast, and `:251` adds KILLED_SELF in addition when the bomb was its own,
    # so a table with both charges a suicide their sum. E25 intended -5/-5 but
    # actually charged -10 for a suicide and -5 for a death by an opponent, which
    # invalidated its arm comparison. With the whole penalty on GOT_KILLED a
    # death is charged once, and the result is identical to the rung-2 table on
    # a board with no opponents, where a suicide also fires both events.
    e.KILLED_SELF: float(os.environ.get("BM_KILLED_SELF", 0)),
    e.GOT_KILLED: float(os.environ.get("BM_GOT_KILLED", -5)),
    # E26 arm H. Zero by default, as in rung 2 and E25.
    # The tested value was 25 rather than 5: the game pays 5:1 kill:coin, and E16
    # set the agent's coin reward to 5, so 25 keeps the game's ratio at this
    # table's scale. It rescales a real game event, so every reward still
    # corresponds to an event that also occurs in the tournament.
    #
    # Why this matters on this rung: the 9 coins are shared four ways, a fair
    # share of ~2.25, and the rung-2 table already collects 2.18 of it. Since
    # score = coins + 5*kills, any further score has to come from kills.
    e.KILLED_OPPONENT: float(os.environ.get("BM_KILL", 0)),
}

# --- Experiment switches --------------------------------------------------
# Read from the environment so a shell loop can sweep seeds and arms without
# editing this file. Every default is the shipped configuration.

# The exploration RNG is seeded with TRAIN_SEED + RUN_INDEX. The shipped table
# came from 106, one of a fifteen-seed sweep (E37), selected on validation world
# seed 550731 and confirmed on held-out 990731. It is the default so the shipped
# configuration needs no environment variables; a sweep overrides it per run.
RUN_INDEX = int(os.environ.get("BM_RUN_INDEX", 106))

# `checkpoints/benedict_task4/q_table_parent.npy`: the previous shipped table,
# copied across digit 8's four danger-row values. It has the same FEATURE_SIZES,
# so `factor` is 1 and this is a row-for-row transfer instead of the
# coarse-to-fine split `warm_start` was written for; the divisibility and layout
# checks still apply. Set BM_WARM="" to start from zero, which discards
# everything earlier rungs learnt about crates and escapes.
WARM_SUFFIX = os.environ.get("BM_WARM", "_parent")
WARM_N = int(os.environ.get("BM_WARM_N", 100))

EPS_DECAY = 0.9995
TRAIN_SEED = 20260731

# Change per experiment. The training log is appended to, so a stale value here
# merges two runs into one file without warning (took half an hour to separate
# again in E05b).
EXPERIMENT = "task4"
RUN_NAME = f"q_{EXPERIMENT}_s{RUN_INDEX}"

# Where this run writes. `MODEL_FILE` is the path callbacks.py loads for play,
# and without BM_MODEL_SUFFIX that is the shipped `q_table.npy`, so training
# with no environment variables would overwrite the tournament model. Training
# writes next to it instead, and replacing the shipped table has to be done by
# hand.
OUTPUT_FILE = MODEL_FILE if os.environ.get("BM_MODEL_SUFFIX") else os.path.join(
    os.path.dirname(MODEL_FILE), "q_table_trained.npy")

# Warm-start parents are stored in `checkpoints/<agent>/`, never next to this
# file: the submission zip is this folder, and 700 MB of checkpoints must not end
# up in it. The path is fixed here and not derived from `MODEL_FILE`'s directory,
# which is only `checkpoints/` when BM_MODEL_SUFFIX is set. Before this, --train
# with no environment variables looked for the parent inside the agent folder
# and crashed.
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                              "checkpoints", AGENT_NAME)


def setup_training(self):
    """Called once before the first round, after `setup` in callbacks.py."""

    self.rng = np.random.default_rng(TRAIN_SEED + RUN_INDEX)
    self.eps = EPS_START
    self.gamma = GAMMA
    self.visits = np.zeros((N_STATES, len(ACTIONS)), dtype=np.int64)
    if WARM_SUFFIX:
        warm_start(self)

    # E20: a diagnostic script that runs BombeRLeWorld with train=True while
    # BM_MODEL_SUFFIX names a real checkpoint overwrites that checkpoint, because
    # MODEL_FILE resolves to its path and the atexit hook below runs on every
    # normal exit. This happened to q_table_e16_c5_k03_s2__ep100000.npy and was
    # only noticed from the file's mtime. We record whether the table existed
    # before this run started, and save_table refuses to write if it did.
    self.model_file_preexisted = os.path.isfile(OUTPUT_FILE)

    # Per-episode accumulators for the learning curve. evaluate.py measures the
    # finished agent; this log shows whether and when training converged.
    self.trainlog = TrainLogger(
        agent=AGENT_NAME,
        run=RUN_NAME,
        # Task-4 training logs go next to the task-4 evaluations; see README.md.
        # TrainLogger resolves a relative out_dir against the repo root, because
        # agents.py changes into this folder around every callback.
        out_dir="results/train/task4_tournament",
        hyperparams={"alpha_exp": ALPHA_EXP,
                     "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay": EPS_DECAY,
                     "gamma": GAMMA,
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
    # this runs on any normal exit. Without it, a round count that is not a
    # multiple of SAVE_EVERY would leave a table up to SAVE_EVERY rounds out of
    # date, which is only noticed when a run fails to reproduce.
    atexit.register(save_table, self)


def save_table(self) -> None:
    """Write the table, unless that would overwrite a checkpoint we did not create.

    A training run overwrites its own file every SAVE_EVERY rounds. The check is
    set once at setup and cleared by the first successful write, so only the
    first write of a run can trigger it, and that is the write that would
    destroy another run's result.

    This raises an error instead of warning. The E18 missing-table fallback only
    warned and continued, and it produced ten evaluations of an all-zero table
    before anyone noticed.
    """

    if self.model_file_preexisted:
        raise FileExistsError(
            f"{OUTPUT_FILE} existed before this run started. Training would "
            "overwrite it. Set BM_MODEL_SUFFIX to a name of this run's own, or "
            "delete the file deliberately if the overwrite is intended."
        )
    # checkpoints/<agent>/ is where BM_MODEL_SUFFIX now points; it is gitignored
    # and may not exist on a fresh clone.
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, self.q)
    write_layout(OUTPUT_FILE)
    self.model_file_preexisted = False


def checkpoint_file(episode: int) -> str:
    """`checkpoints/<agent>/q_table<suffix>__ep<N>.npy`, the same path that
    `BM_MODEL_SUFFIX=<suffix>__ep<N>` resolves to. A checkpoint can therefore be
    evaluated by setting that one variable, and `callbacks.py` needs no special
    case.

    Always under `checkpoints/`, never next to this file: the submission zip is
    this folder, and the 700 MB of rung-2 tables were moved out of it for this
    reason.
    """

    base, ext = os.path.splitext(os.path.join(CHECKPOINT_DIR,
                                              os.path.basename(OUTPUT_FILE)))
    return f"{base}__ep{episode}{ext}"


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """One Q-learning update per step."""

    s = state_to_features(old_game_state) if old_game_state is not None else None
    reward = reward_from_events(self, events)

    # Tally before the guard: the first call of a round has no old state to
    # learn from, but its events still happened and belong in the episode total.
    self.episode_events.extend(events)
    self.episode_reward += reward

    if old_game_state is None:
        return

    s_next = state_to_features(new_game_state)
    a = ACTIONS.index(self_action)

    td_target = reward + self.gamma * np.max(self.q[s_next])

    td_error = td_target - self.q[s, a]
    apply_update(self, s, a, td_error)

    self.episode_td.append(abs(td_error))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Terminal update, model save, and one row in the training log."""

    s = state_to_features(last_game_state)
    a = ACTIONS.index(last_action)
    reward = reward_from_events(self, events)

    td_target = reward

    td_error = td_target - self.q[s, a]
    apply_update(self, s, a, td_error)

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

    A `.npy` stores how many rows it has but not what they mean. Two different
    `FEATURE_SIZES` can give the same row count, so from the table alone a warm
    start cannot tell whether it is a valid parent (see the append-only argument
    in `warm_start`). The layout is stored as one line of JSON.
    """

    with open(table_file + LAYOUT_EXT, "w") as fh:
        json.dump({"feature_sizes": list(FEATURE_SIZES)}, fh)


def read_layout(table_file: str) -> list | None:
    """The layout a table was trained on, or None for tables written before
    layouts were recorded, including the E16 parent the shipped model starts
    from."""

    try:
        with open(table_file + LAYOUT_EXT) as fh:
            return json.load(fh)["feature_sizes"]
    except (OSError, KeyError, ValueError):
        return None


def warm_start(self) -> None:
    """Initialise this table from a coarser one, each row from its parent.

    E20 measured the problem this addresses: the distance digit splits every old
    row into five, and on two of five seeds the run filled only ~485 of its ~840
    used rows, so the 64 000-row map held the information of a 12 800-row one.
    The coarse table already knows what those states are worth, apart from the
    distinction the new digit adds, so each child row starts from its parent
    instead of from zero.

    New digits must be appended, never inserted. `encode` is mixed radix, so
    appending digits to `FEATURE_SIZES` multiplies every old index by the new
    radices: parent row i becomes children `k*i .. k*i+k-1`, which is what
    `np.repeat` produces. If a digit is inserted anywhere else, the row count
    is the same and the divisibility check below still passes, but the mapping
    is wrong without any error. On 5 000 random states an inserted digit
    mis-maps 39 % of them. This is why the layout is saved next to every table
    and checked here.

    Only cells whose parent has a nonzero value get the pseudo-count. Giving it
    to the others would start new rows at a twenty-fifth of their learning rate
    with no benefit.
    """

    coarse_file = os.path.join(CHECKPOINT_DIR, f"q_table{WARM_SUFFIX}.npy")
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
    """Per-cell learning rate. See the module docstring for why this matters.

    """
    self.visits[s, a] += 1
    return 1.0 / self.visits[s, a] ** ALPHA_EXP


def apply_update(self, s: int, a: int, td_error: float) -> None:
    """One TD update."""
    self.q[s, a] += learning_rate(self, s, a) * td_error


def decode_row(idx: int) -> tuple:
    """Row index -> digit tuple. Inverse of `callbacks.encode`; digit 0 is the
    most significant, matching the mixed-radix encoding there."""

    digits = []
    for size in reversed(FEATURE_SIZES):
        digits.append(idx % size)
        idx //= size
    return tuple(reversed(digits))


def reward_from_events(self, events: List[str]) -> float:
    return sum(REWARDS.get(ev, 0.0) for ev in events) + STEP_COST

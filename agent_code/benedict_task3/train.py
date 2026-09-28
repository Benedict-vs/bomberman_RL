"""Q-learning updates for `benedict_task3`.

Loaded only with `--train`, so nothing here runs in the tournament. That is why
`tools/` may be imported (defensively) and why the exploration RNG lives here.

**Nothing in this file produced the shipped `q_table.npy`, and that is the result
of rung 3.** The shipped table is the *rung-2* table, byte for byte -- rebuild it
with `agent_code/benedict_task2/train.py` at its defaults. Rung 3 is won by a
feature change (`BM_HUNT`, see `callbacks.py`) read through that frozen policy.

Twenty-five training runs across E25-E27 -- four reward configurations, two
feature maps, five seeds each -- and **not one beat the untrained table**. The
cause is known and is written up in `experiments/benedict_task3.md` §3: the
reward table was calibrated on a board the agent had to itself, where it earned
77.3 against 40.1 of step cost. Sharing the board with three opponents cuts
earnings ~4x while steps alive fall only 1.85x, so the same table pays 0.7:1, an
action-independent constant dominates the return, and the fixed point becomes
action-independent too. E27 confirmed it (raising `BM_CRATE` to 1.0 is worth
+0.93 score and +11.5 crates, paired, and moves the decision margin from 0.0003
to 0.05) and still finished 1.2 points behind not training at all.

So this file is kept for the record and for rung 4, not because it is on the path
to the shipped model. If you train with it, read `experiments/benedict_task3.md`
first.

**Reproducibility is weaker here than on rung 2, in both directions.** `main.py`
does not seed the provided agents and each reseeds the global RNG from OS entropy
in `setup`, which runs after ours -- so nothing here can precede it, and two runs
of the same command give different tables. Compare arms across seeds, never on a
single run. Evaluations are only *partly* reproducible too: `tools/evaluate.py`
seeds the opponents' `np.random`, but `coin_collector_agent` and
`rule_based_agent` also shuffle with the *stdlib* `random`, which that does not
touch -- 22.7 % of rounds repeat exactly, the means repeat to four decimals
(`MEASUREMENT.md`). Rung-3 pairing is on arenas only.

Three things that are easy to get wrong if you do train with this file:

- **The world seed 810731 is not the evaluation seed.** Never train on 20260731
  or 550731 -- the agent would then be measured on arenas it trained on.
- **`BM_HUNT=0` is required to touch anything of rung-2 lineage** (the warm-start
  parent, an E24/E25 table). Since E26 the feature is on by default and changes
  what digits 6 and 7 mean; the row count is identical either way, so a mismatch
  produces a wrong table rather than an error. Coarse-to-fine parent rebuilding
  lives in `agent_code/benedict_task2/train.py`, where that lineage belongs.
- **`WARM_SUFFIX` warm-starts from `q_table_rung2ship`** -- same layout, so
  `warm_start` transfers row for row at `factor = 1`. E27 measured that starting
  cold is neither better nor worse (1.013 vs 1.393), so the warm start is a
  convenience, not the cause of anything.

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
KILLED_SELF         0          E26. Was -5 through rungs 1-2, where it removed every
                               suicide at once. Now 0 because GOT_KILLED already
                               fires on a suicide (environment.py:251 adds
                               KILLED_SELF *on top* of :264's GOT_KILLED), so
                               carrying both double-priced it. E25 shipped -5/-5
                               believing it symmetric and actually paid -10 for a
                               suicide, which voided its arm contrast.
GOT_KILLED         -5          E26. Prices death exactly once, whatever killed the
                               agent. Provably identical to the rung-2 table on a
                               board with no opponents, since a suicide fires both
                               events there too -- so this is a correctness fix,
                               not a retuning.
KILLED_OPPONENT     0          E26 arm H sets 25, preserving the game's own 5:1
                               kill:coin ratio at this table's scale (E16 put the
                               agent's coin at 5). Coins are saturated on rung 3 --
                               9 shared four ways is a ~2.25 fair share and the
                               rung-2 table already banks 2.18 -- and
                               score = coins + 5*kills, so every further point has
                               to come from kills.
STEP_COST          -0.1        Shortest-path pressure; the round is capped at 400
                               steps and 99 % of rounds hit that cap. A switch
                               since E27 (BM_STEP_COST): it is action-INDEPENDENT,
                               so once it dominates the return the fixed point is
                               action-independent too. Rung 2 earned 77.3 against
                               40.1 of it; rung 3 earns 18.3 against 26.1 from the
                               same table, which is the rung-3 collapse.
WARM_N             100         E23. Not optional: alpha is exactly 1 on a cell's
                               first update, so an untouched transfer is overwritten
                               immediately. 10 000 and 100 000 are both worse -- the
                               table then cannot differentiate the rows the new digit
                               created.
episodes           20 000      E23 on rung 2: longer is worse there. NOT true on
                               rung 3 -- E27 measured every arm flat or improving
                               from 20 000 to 40 000 (C10 retained 125 %).
=================  ==========  ====================================================

Every row above is a *rung-2* justification. E27 re-tested three of them in the
opponent field and the rung-2 answer held for two: alpha (1/N^0.55 is worse,
0.987 vs 1.393) and the warm start (cold is 1.013, indistinguishable). The one
that does not transfer is the reward *scale* -- see the module docstring.
=================  ==========  ====================================================
"""

import atexit
import json
import os
from typing import List

import numpy as np

import events as e
from .callbacks import (state_to_features, ACTIONS, MODEL_FILE, N_STATES,
                        DELTAS, FEATURE_SIZES, ABLATE)

try:
    from tools.trainlog import TrainLogger
except ImportError:     # tools/ is not part of the submission
    TrainLogger = None


AGENT_NAME = "benedict_task3"

# E27. An environment switch because it is one of two knobs on the same quantity:
# the value function's dynamic range is gross earnings against this cost, and the
# cost is *action-independent*, so once it dominates, the fixed point is too.
# Rung 2 earned 77.3 against 40.1 (ratio 1.93); rung 3 earns 18.3 against 26.1
# (0.70) from the identical table, because nine coins shared four ways cuts
# earnings ~4x while steps alive fall only 1.85x. BM_CRATE=1.0 and
# BM_STEP_COST=-0.03 reach a healthy ratio from opposite directions.
STEP_COST = float(os.environ.get("BM_STEP_COST", -0.1))
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
    # E26 correctness fix. `environment.py:264` adds GOT_KILLED to *every* agent
    # killed by a blast and `:251` adds KILLED_SELF **on top** when the bomb was
    # its own -- so a table carrying both prices a suicide at their sum. E25
    # intended -5/-5 and actually paid -10 for a suicide and -5 for an opponent's
    # kill, which voided its arm contrast. Putting the whole penalty on
    # GOT_KILLED prices death exactly once, and is *identical* to the rung-2
    # table on a board with no opponents, where a suicide fires both events too.
    e.KILLED_SELF: float(os.environ.get("BM_KILLED_SELF", 0)),
    e.GOT_KILLED: float(os.environ.get("BM_GOT_KILLED", -5)),
    # E26 arm H. Zero by default, which is what rung 2 and E25 both used.
    # 25 rather than 5: the game pays 5:1 kill:coin, and E16 put the agent's coin
    # at 5, so 25 preserves the game's own ratio at this table's scale. It is a
    # rescaling of a real game event, not an invented one -- the reward table
    # stays a map of events the tournament also generates.
    #
    # Why it matters on this rung: the 9 coins are shared four ways, a ~2.25 fair
    # share, and the rung-2 table already banks 2.18 of it. score = coins + 5*kills,
    # so every further point of score has to come from kills.
    e.KILLED_OPPONENT: float(os.environ.get("BM_KILL", 0)),
}

# --- Experiment switches --------------------------------------------------
# Read from the environment so a shell loop can sweep seeds and arms without
# editing this file. Every default is the shipped configuration.

# Seeds the exploration RNG as TRAIN_SEED + RUN_INDEX. 5 is the seed the shipped
# table came from, selected on the held-out world seed 550731 (E23); 0-4 are the
# sweep it was selected against.
RUN_INDEX = int(os.environ.get("BM_RUN_INDEX", 20))

ALPHA_MODE = os.environ.get("BM_ALPHA", "visit")    # "visit" | "const" (E06)
EPS_MODE = os.environ.get("BM_EPS", "decay")        # "decay" | "const" (E05)

# E19, rejected: potential-based shaping, F = gamma*Phi(s') - Phi(s) with
# Phi = -BM_SHAPE * (BFS distance to digit 6's goal). Kept at its default of 0 --
# which skips the whole mechanism, BFS included -- so E19 stays reproducible.
# It fails because a row of this table is a bucket of states with different Phi,
# so the offset does not cancel between actions the way the theorem needs.
SHAPE = float(os.environ.get("BM_SHAPE", 0))

# E25: the shipped rung-2 table, copied to checkpoints/benedict_task3/. Same
# FEATURE_SIZES, so `factor` is 1 and warm_start is a row-for-row transfer rather
# than the coarse-to-fine split it was built for -- the divisibility and layout
# checks both still apply. Empty starts from zero, which throws away everything
# rung 2 learnt about crates and escapes.
WARM_SUFFIX = os.environ.get("BM_WARM", "_rung2ship")
WARM_N = int(os.environ.get("BM_WARM_N", 100))

# Set with BM_ABLATE=target_dist to also write the coarse parent this run is
# really producing -- see `write_parent`. Off by default; it is only meaningful
# for that one ablation arm and costs an extra file per save otherwise.
SAVE_PARENT = os.environ.get("BM_SAVE_PARENT", "") not in ("", "0")

EPS_DECAY = 0.9995 if EPS_MODE == "decay" else 1.0
TRAIN_SEED = 20260731

# Change per experiment. The training log is *appended* to, so a stale value here
# silently merges two runs into one file (cost half an hour to unpick in E05b).
EXPERIMENT = "e27"
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
        # Rung-2 logs go beside the rung-2 evaluations; see README.md.
        # TrainLogger anchors a relative out_dir to the repo root -- agents.py
        # chdirs into this folder around every callback, so the cwd is not it.
        out_dir="results/train/task3_opponents",
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
    if SAVE_PARENT:
        write_parent(self)
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


def write_parent(self) -> None:
    """Also write the *coarse* table an ablated run is really producing.

    `BM_ABLATE=target_dist` pins the trailing digit to 0, and because that digit
    is least significant, pinning it is a bijective relabeling: the coarse row i
    lands at row k*i and the other k-1 rows of each group are never touched
    (E20 finding 2, verified here -- they hold exactly 0.0). So `q[::k]` *is* the
    table the pre-E20 feature map would have produced.

    Writing it out removes the last reason to check out an old commit: the
    warm-start parent can be rebuilt at HEAD. Confirmed by training 20 000
    episodes this way and comparing against the committed
    `q_table_e16_c5_k03_s0__ep20000.npy` -- identical.

    The sidecar records the *parent's* layout, not this run's, which is what
    lets `warm_start` accept the result as a legitimate ancestor.
    """

    if ABLATE != "target_dist":
        raise ValueError(
            "BM_SAVE_PARENT only makes sense with BM_ABLATE=target_dist. "
            f"With BM_ABLATE={ABLATE!r} the trailing digit is live, so the "
            "stride-k rows are not a coarse table -- they are every fifth row "
            "of a finer one, which is not the same thing and would warm-start "
            "into nonsense.")

    stride = FEATURE_SIZES[-1]
    base, ext = os.path.splitext(MODEL_FILE)
    out = f"{base}__coarse{ext}"
    np.save(out, self.q[::stride])
    with open(out + LAYOUT_EXT, "w") as fh:
        json.dump({"feature_sizes": list(FEATURE_SIZES[:-1])}, fh)


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

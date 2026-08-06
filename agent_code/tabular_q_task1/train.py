"""Q-learning updates for `tabular_q_task1`.

Loaded only with `--train`, so nothing here runs in the tournament. That is why
`tools/` may be imported (defensively) and why the exploration RNG lives here.

Settings inherited from task 1, with the evidence (`experiments/task1.md`):

- **alpha = 1/N(s,a)^0.7, not a constant.** Largest single effect measured on
  rung 1: constant alpha gave 20.78 +- 6.98 coins over 5 seeds, per-cell gave
  49.15 +- 1.23. A constant alpha satisfies neither of L26's convergence
  conditions, so busy cells never settle -- and an unsettled cell here is not a
  small error but an absorbing deadlock.
- **KILLED_SELF = -5.** Removed every suicide at once, which is why BOMB can
  stay in the action set instead of being masked out.
- **The epsilon schedule is not justified by rung-1 numbers** (no demonstrated
  difference over 5 seeds). Kept for task 2, where danger states are a distinct
  region of the state space that only a surviving agent ever reaches.

Reproducibility needs BOTH the exploration seed here and `main.py --seed`;
either alone leaves runs incomparable. Train on a world seed that is *not* the
evaluation seed (20260731), or the agent is measured on arenas it trained on.
"""

import os
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS, MODEL_FILE, N_STATES

try:
    from tools.trainlog import TrainLogger
except ImportError:     # tools/ is not part of the submission
    TrainLogger = None


AGENT_NAME = "tabular_q_task1"

STEP_COST = -0.1    # encourages shorter paths
GAMMA = 0.9

ALPHA = 0.1         # only used when ALPHA_MODE == "const"
ALPHA_EXP = 0.7     # in (0.5, 1]: where sum(a)=inf and sum(a^2)<inf both hold (L26)

EPS_START = 0.2
EPS_END = 0.02

REWARDS = {
    e.COIN_COLLECTED: 5,
    e.INVALID_ACTION: -1,
    e.WAITED: -0.1,
    e.KILLED_SELF: -5,
    # TODO task 2: CRATE_DESTROYED, and finally ablate the coin reward against
    # the game's actual +1 -- open since the first experiment.
}

# --- Experiment switches --------------------------------------------------
# Read from the environment so a shell loop can sweep seeds and arms without
# editing this file. Defaults are the settings task 1 ended on.
RUN_INDEX = int(os.environ.get("BM_RUN_INDEX", 0))
ALPHA_MODE = os.environ.get("BM_ALPHA", "visit")    # "visit" | "const"
EPS_MODE = os.environ.get("BM_EPS", "decay")        # "decay" | "const"

EPS_DECAY = 0.9995 if EPS_MODE == "decay" else 1.0
TRAIN_SEED = 20260731

EXPERIMENT = "merged"   # change per experiment; the training log is appended to
RUN_NAME = f"q_{EXPERIMENT}_s{RUN_INDEX}"


def setup_training(self):
    """Called once before the first round, after `setup` in callbacks.py."""

    self.rng = np.random.default_rng(TRAIN_SEED + RUN_INDEX)
    self.eps = EPS_START
    self.gamma = GAMMA
    self.visits = np.zeros((N_STATES, len(ACTIONS)), dtype=np.int64)

    # Per-episode accumulators for the learning curve. evaluate.py measures the
    # finished agent; this is what shows whether it converged, and when.
    self.trainlog = TrainLogger(
        agent=AGENT_NAME,
        run=RUN_NAME,
        hyperparams={"alpha": ALPHA, "alpha_mode": ALPHA_MODE, "alpha_exp": ALPHA_EXP,
                     "eps_start": EPS_START, "eps_mode": EPS_MODE, "eps_end": EPS_END,
                     "eps_decay": EPS_DECAY,
                     "gamma": GAMMA,
                     "train_seed": TRAIN_SEED + RUN_INDEX, "run_index": RUN_INDEX,
                     "step_cost": STEP_COST,
                     "rewards": {k: v for k, v in REWARDS.items()},
                     "n_states": len(self.q),
                     "features": "4 wall bits + BFS direction to nearest coin"},
        extra_columns=["td_error"],
    ) if TrainLogger else None
    self.episode_events = []
    self.episode_reward = 0.0
    self.episode_td = []


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

    td_target = reward + self.gamma * np.max(self.q[s_next])
    td_error = td_target - self.q[s, a]
    self.q[s, a] += learning_rate(self, s, a) * td_error

    self.episode_td.append(abs(td_error))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Terminal update, model save, and one row in the training log."""

    s = state_to_features(last_game_state)
    a = ACTIONS.index(last_action)
    reward = reward_from_events(self, events)

    # Q(terminal, .) = 0, so no bootstrap term. Leaving it in would teach the
    # agent that dying is as good as surviving -- step (0) of L26's algorithm.
    td_target = reward
    td_error = td_target - self.q[s, a]
    self.q[s, a] += learning_rate(self, s, a) * td_error

    np.save(MODEL_FILE, self.q)

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


def learning_rate(self, s: int, a: int) -> float:
    """Per-cell learning rate. See the module docstring for why this matters."""

    self.visits[s, a] += 1
    if ALPHA_MODE == "visit":
        return 1.0 / self.visits[s, a] ** ALPHA_EXP
    return ALPHA


def reward_from_events(self, events: List[str]) -> float:
    return sum(REWARDS.get(ev, 0.0) for ev in events) + STEP_COST

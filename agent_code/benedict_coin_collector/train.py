import os
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS, MODEL_FILE, N_STATES

try:
    from tools.trainlog import TrainLogger
except ImportError:     # tools/ is not part of the submission
    TrainLogger = None


# # Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# # Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"

STEP_COST = -0.1  # cost of taking a step, to encourage shorter paths

ALPHA = 0.1         # learning rate
GAMMA = 0.9         # discount factor
EPS_START = 0.2           # exploration rate during training
EPS_END = 0.02

REWARDS = {
    e.COIN_COLLECTED: 5,
    e.INVALID_ACTION: -1,
    e.WAITED: -0.1,
    e.KILLED_SELF: -5,
}

# --- Reproducibility ------------------------------------------------------
# `--seed` fixes the arenas but NOT the agent's exploration, so two training
# runs of the same code diverge. That made v4 and v4b incomparable (a 17-coin
# swing from one wrong cell), so the exploration RNG is seeded here.
#
# Seeding alone only buys reproducibility, not reliability: a bad draw comes
# back identically. Reliability comes from running the SAME configuration under
# several indices and reporting the spread -- hence BM_RUN_INDEX, so a shell
# loop can do that without editing this file. Absent (i.e. in the tournament,
# where train.py is never imported at all) it defaults to 0.
TRAIN_SEED = 20260731
RUN_INDEX = int(os.environ.get("BM_RUN_INDEX", 0))

ALPHA_MODE = os.environ.get("BM_ALPHA", "const")   # "const" | "visit"
ALPHA_EXP = 0.7      # in (0.5, 1]: where sum(a)=inf and sum(a^2)<inf both hold (L26)

EPS_MODE = os.environ.get("BM_EPS", "decay")          # "decay" | "const"
EPS_DECAY = 0.9995 if EPS_MODE == "decay" else 1.0

RUN_NAME = f"q_e07c_s{RUN_INDEX}_task1"               # per-cell alpha, constant eps

# # Change this per experiment: the training log is appended to, not overwritten.
# RUN_NAME = f"q_e06{'b' if ALPHA_MODE == 'visit' else 'a'}_s{RUN_INDEX}_task1"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.rng = np.random.default_rng(TRAIN_SEED + RUN_INDEX)
    self.eps = EPS_START
    self.gamma = GAMMA
    
    self.visits = np.zeros((N_STATES, len(ACTIONS)), dtype=np.int64)

    # Per-episode accumulators for the learning curve. evaluate.py measures the
    # finished agent; this is what shows whether it converged, and when.
    self.trainlog = TrainLogger(
        agent="benedict_coin_collector",
        run=RUN_NAME,
        hyperparams={"alpha": ALPHA, "alpha_mode": ALPHA_MODE, "alpha_EXP": ALPHA_EXP,
                     "eps_start": EPS_START, "eps_mode": EPS_MODE, "eps_end": EPS_END, "eps_decay": EPS_DECAY,
                     "gamma": GAMMA,
                     "train_seed": TRAIN_SEED + RUN_INDEX, "run_index": RUN_INDEX,
                     "step_cost": STEP_COST,
                     "rewards": {k: v for k, v in REWARDS.items()},
                     "n_states": len(self.q),
                     "features": "4 wall bits + dx, dy clipped to [-3, 3]"},
        extra_columns=["td_error"],
    ) if TrainLogger else None
    self.episode_events = []
    self.episode_reward = 0.0
    self.episode_td = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

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
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)
    
    s = state_to_features(last_game_state)
    a = ACTIONS.index(last_action)
    reward = reward_from_events(self, events)

    td_target = reward                  # no next state, so no future reward
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
    self.eps = max(EPS_END, self.eps * EPS_DECAY)  # decay exploration rate per episode


def reward_from_events(self, events: List[str]) -> float:
    return sum(REWARDS.get(ev, 0.0) for ev in events) + STEP_COST

def learning_rate(self, s: int, a: int) -> float:
    self.visits[s, a] += 1
    if ALPHA_MODE == "visit":
        return 1.0 / self.visits[s, a] ** ALPHA_EXP
    return ALPHA
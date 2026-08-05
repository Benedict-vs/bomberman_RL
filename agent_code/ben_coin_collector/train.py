import pickle
from typing import List

import events as e

from .callbacks import MODEL_FILE
from .features import (ACTIONS, distance_to_nearest_coin, state_to_features)


try:
    from tools.trainlog import TrainLogger
except ImportError:
    TrainLogger = None


ALPHA = 0.15
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
POTENTIAL_SCALE = 0.1

RUN_NAME = "ben_coin_collector_Qlearning_potential_scale01_coin15_invalid10_test6"


STEP_PENALTY = -0.01

EVENT_REWARDS = {
    e.COIN_COLLECTED: 15.0,
    e.INVALID_ACTION: -10.0,
    e.WAITED: -0.02,
    
}

def coin_potential(game_state):
    #Berechnet das Zustandspotential anhand der BFS-Distanz

    if game_state is None:
        return 0.0

    distance = distance_to_nearest_coin(
        field=game_state["field"],
        start=game_state["self"][3],
        coins=game_state["coins"],
    )

    if distance is None:
        return 0.0

    return 1.0 / (distance + 1)


def potential_shaping_reward(old_game_state, new_game_state):
    #Berechnet den zusätzlichen potentialbasierten Reward

    old_potential = coin_potential(old_game_state)
    new_potential = coin_potential(new_game_state)

    return POTENTIAL_SCALE * (
        GAMMA * new_potential - old_potential
    )


def setup_training(self):
    #Initialisiert die Variablen für das Training

    self.epsilon = EPSILON_START
    self.episode_events = []
    self.episode_reward = 0.0

    if TrainLogger is not None:
        self.trainlog = TrainLogger(
            agent="ben_coin_collector",
            run=RUN_NAME,
            hyperparams={
                "alpha": ALPHA,
                "gamma": GAMMA,
                "epsilon_start": EPSILON_START,
                "epsilon_end": EPSILON_END,
                "epsilon_decay": EPSILON_DECAY,
                "reward_variant": "potential_shaping",
                "potential_scale": POTENTIAL_SCALE,
            },
            extra_columns=["states"],
        )
    else:
        self.trainlog = None


def update_q(self, state, action, reward, next_state):
    #Aktualisiert einen Q-Wert anhand eines Übergangs

    if state is None:
        return

    if action not in ACTIONS:
        return

    action_index = ACTIONS.index(action)
    old_value = self.q_table[state][action_index]

    if next_state is None:
        continuation = 0.0
    else:
        continuation = GAMMA * self.q_table[next_state].max()

    target = reward + continuation
    learning_error = target - old_value

    self.q_table[state][action_index] = (
        old_value + ALPHA * learning_error
    )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    #Verarbeitet einen Übergang innerhalb einer laufenden Runde

    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    event_reward = reward_from_events(self, events)
    shaping_reward = potential_shaping_reward(
        old_game_state=old_game_state,
        new_game_state=new_game_state,)
    reward = event_reward + shaping_reward

    update_q(
        self=self,
        state=old_state,
        action=self_action,
        reward=reward,
        next_state=new_state,
    )

    self.episode_events.extend(events)
    self.episode_reward += reward


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    #Verarbeitet das Rundenende und speichert die Q-Tabelle

    event_reward = reward_from_events(self, events)
    shaping_reward = potential_shaping_reward(
        old_game_state=last_game_state,
        new_game_state=None,
)

    final_reward = event_reward + shaping_reward

    update_q(
        self=self,
        state=state_to_features(last_game_state),
        action=last_action,
        reward=final_reward,
        next_state=None,
    )

    self.episode_events.extend(events)
    self.episode_reward += final_reward

    self.epsilon = max(
        EPSILON_END,
        self.epsilon * EPSILON_DECAY,
    )

    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(dict(self.q_table), model_file)

    if self.trainlog is not None:
        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=last_game_state["self"][1],
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={
                "states": len(self.q_table),
            },
        )

    self.episode_events = []
    self.episode_reward = 0.0


def reward_from_events(self, events: List[str]) -> float:
    #Berechnet den Reward eines Spielschritts

    reward = STEP_PENALTY

    for event in events:
        reward += EVENT_REWARDS.get(event, 0.0)

    self.logger.debug(
        "Reward %.3f for events %s.",
        reward,
        events,
    )

    return reward
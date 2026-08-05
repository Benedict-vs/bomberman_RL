from collections import namedtuple, deque

import os
import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS, ALLOWED_IDX
#import trainlogger if available, else set to None
try:
    from tools.trainlog import TrainLogger
except ImportError:
    TrainLogger = None

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

## hyperparameters
ALPHA = 0.1       # lerining rate (
GAMMA = 0.9       # discount factor for futere rewards
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997   # 0.997^1000 ~ 0.05, erreicht EPSILON_END innerhalb eines 1000-Runden-Laufs
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
#PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.

    """
    self.epsilon = EPSILON_START
    #logging trainin process
    self.trainlog = TrainLogger(
        agent="maxi_coin_collector", 
        run=os.environ.get("MAXI_RUN", "q_v2_task1"),
        hyperparams={"alpha": ALPHA, "gamma": GAMMA, "eps_decay": EPSILON_DECAY},
        extra_columns=["td_error", "table_size"],
    ) if TrainLogger else None
    
    self.episode_events = []
    self.episode_reward = 0.0
    self.losses = [] 
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    update_q_table(self, old_game_state, self_action, new_game_state, events)

    reward = reward_from_events(self, events)
    self.episode_events.extend(events)
    self.episode_reward += reward

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    #last q update for the last step of the game, where next_state is None
    update_q_table(self, last_game_state, last_action, None, events)

    # Epsilon decay, reduce exploration over time

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

   
    reward = reward_from_events(self, events)
    self.episode_events.extend(events)
    self.episode_reward += reward
    
    if self.trainlog:
        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=last_game_state["self"][1],
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={
                "td_error": float(np.mean(self.losses)) if self.losses else 0.0,
                "table_size": len(self.model)
            },
        )
    self.episode_events = []
    self.episode_reward = 0.0
    self.losses = []

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def update_q_table(self, old_game_state, action, new_game_state, events):
    """
    q learning steps with bellmann equation
    """
    # in der ersten Runde bzw. wenn der Agent nie gehandelt hat, gibt es nichts zu lernen
    if old_game_state is None or action is None:
        return

    old_features = state_to_features(old_game_state)
    reward = reward_from_events(self, events)
    action_idx = ACTIONS.index(action)

    # making sure old state is in the model, if not initialize with zeros
    if old_features not in self.model:
        self.model[old_features] = np.zeros(len(ACTIONS))

    # maximum future Q-value for the next state, if next state is None (game over), set to 0
    if new_game_state is None:
        max_future_q = 0.0 # game ended
    else:
        new_features = state_to_features(new_game_state)
        if new_features not in self.model:
            self.model[new_features] = np.zeros(len(ACTIONS))
        # max nur über die erlaubten Aktionen -- sonst bootstrappt das Update auf
        # einen Q-Wert, den die Politik nie wählen kann
        max_future_q = max(self.model[new_features][i] for i in ALLOWED_IDX)

    #get current q value
    current_q = self.model[old_features][action_idx]

    # calculate td error 
    td_error = reward + GAMMA * max_future_q - current_q
    self.losses.append(td_error)

    # update q value using bellmann equation
    self.model[old_features][action_idx] = current_q + ALPHA * td_error


def reward_from_events(self, events: List[str]) -> int:
    """
   rewards for coin collector; collecting and punishing invalid actions and waiting
    """
    game_rewards = {
        e.COIN_COLLECTED: 5.0,   # 
        e.WAITED: -0.5,           # Verhindert Rumsitzen
        e.INVALID_ACTION: -1.0,   # Gegen die Wand laufen ist schlecht
    }
    
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            
    # small negative reward for each step to encourage faster coin collection (disabled for first run)
    #reward_sum -= 0.1 
    
    return reward_sum
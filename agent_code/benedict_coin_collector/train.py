import numpy as np
from typing import List
import events as e
from .callbacks import state_to_features, ACTIONS, MODEL_FILE


# # Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# # Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"

STEP_COST = -0.1  # cost of taking a step, to encourage shorter paths

ALPHA = 0.1         # learning rate
GAMMA = 0.9         # discount factor

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.eps = 0.2
    self.alpha = ALPHA
    self.gamma = GAMMA
    

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

    if old_game_state is None:
        return
    
    s = state_to_features(old_game_state)
    s_next = state_to_features(new_game_state)
    a = ACTIONS.index(self_action)
    reward = reward_from_events(self, events)
    
    td_target = reward + self.gamma * np.max(self.q[s_next])
    self.q[s, a] += self.alpha * (td_target - self.q[s, a])

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
    self.q[s, a] += self.alpha * (td_target - self.q[s, a])

    np.save(MODEL_FILE, self.q)

def reward_from_events(self, events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.1,
    }
    return sum(rewards.get(ev, 0.0) for ev in events) + STEP_COST

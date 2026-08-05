import os
import pickle
import random

import numpy as np
from .features import state_to_features


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Stufe 1 (coin-heaven): keine Kisten, also hat eine Bombe keinen Nutzen -- und ohne
# Gefahren-Feature ("liege ich im Explosionsradius?") ist Weglaufen gar nicht lernbar,
# jede gelegte Bombe endet im Selbstmord. WAIT ist hier reiner Zeitverlust.
# Die Q-Vektoren bleiben trotzdem 6 lang, damit ab Stufe 2 nur diese Liste wächst.
ALLOWED_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
ALLOWED_IDX = [ACTIONS.index(a) for a in ALLOWED_ACTIONS]


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Q-Tabelle: feature-tuple -> np.array mit einem Q-Wert je Aktion
        self.model = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)

    #epsilon value from train.py, if not training set to 0.0, so no exploration
    epsilon = getattr(self, 'epsilon', 0.0) if self.train else 0.0
    
    #exploration 
    if self.train and random.random() < epsilon:
        self.logger.debug("Exploration: Choosing random action.")
        return random.choice(ALLOWED_ACTIONS)

    #if first time seeing state, initialize Q-values for all actions to 0
    if features not in self.model:
        self.model[features] = np.zeros(len(ACTIONS))

    #exploitation, nur über die auf dieser Stufe erlaubten Aktionen
    q_values = self.model[features]

    #if tie between multiple actions, randomly choose one of the best actions
    max_q = max(q_values[i] for i in ALLOWED_IDX)
    best_actions = [i for i in ALLOWED_IDX if q_values[i] == max_q]
    chosen_action_index = random.choice(best_actions)


    self.logger.debug("Querying model for action.")
    return ACTIONS[chosen_action_index]



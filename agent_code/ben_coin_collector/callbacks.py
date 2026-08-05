#Aktionsauswahl des tabellarischen Q-Learning-Agenten

import os
import pickle
import random

from collections import defaultdict
from .features import ACTIONS, state_to_features

import numpy as np


MODEL_FILE = "my-saved-model.pt"


def new_q_row():
    #Erzeugt Q-Werte für einen bisher unbekannten Zustand

    return np.zeros(len(ACTIONS), dtype=float)


def setup(self):
    #Erstellt eine neue Q-Tabelle oder lädt eine gespeicherte Tabelle

    self.q_table = defaultdict(new_q_row)

    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as model_file:
            saved_table = pickle.load(model_file)

        self.q_table.update(saved_table)

        self.logger.info(
            "Loaded Q-table with %d states.",
            len(self.q_table),
        )

    elif self.train:
        self.logger.info(
            "No saved model found. Starting with an empty Q-table."
        )

    else:
        self.logger.warning(
            "No saved model found. Actions are based on zero Q-values."
        )


def act(self, game_state: dict) -> str:
    #Wählt während des Trainings epsilon-greedy eine Aktion aus

    state = state_to_features(game_state)

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Exploring in state %s.", state)
        return str(np.random.choice(ACTIONS))

    q_values = self.q_table[state]
    maximum = q_values.max()

    best_indices = np.flatnonzero(q_values == maximum)
    chosen_index = int(np.random.choice(best_indices))

    action = ACTIONS[chosen_index]

    self.logger.debug(
        "State %s, Q-values %s, action %s.",
        state,
        q_values,
        action,
    )

    return action

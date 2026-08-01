"""Modell A — tabellarisches Q-Learning.

Immer geladen, auch im Turnier. Alles, was nur zum Trainieren gebraucht wird,
steht in `train.py`.
"""

import os
import pickle
import random
from collections import defaultdict

import numpy as np

from .features import ACTIONS, state_to_features

# Relativer Pfad — `agents.py:305` wechselt vor jedem Callback in das
# Agentenverzeichnis, die Datei landet also neben diesem Modul. Absolute Pfade
# sind laut AGENTS.md der klassische Abgabe-Crash.
MODEL_FILE = "my-saved-model.pt"


def new_q_row():
    """Q-Werte eines noch nie besuchten Zustands.

    Als benannte Funktion statt Lambda, damit `defaultdict` picklebar bleibt.
    Null-Initialisierung ist bei negativer Schrittstrafe optimistisch: unbesuchte
    Aktionen sehen besser aus als besuchte, das treibt die Exploration zusätzlich
    zu epsilon.
    """
    return np.zeros(len(ACTIONS))


def setup(self):
    """Q-Tabelle laden oder leer anlegen. Einmal pro Agent, vor der ersten Runde."""
    self.q_table = defaultdict(new_q_row)

    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as file:
            self.q_table.update(pickle.load(file))
        self.logger.info(f"Loaded Q-table with {len(self.q_table)} states.")
    elif self.train:
        self.logger.info("No model found — starting from an empty Q-table.")
    else:
        # Kein Absturz, aber im Turnier wäre das ein Totalausfall.
        self.logger.warning(f"No {MODEL_FILE} found and not training — acting at random.")


def act(self, game_state: dict) -> str:
    """epsilon-greedy über die Q-Tabelle; greedy sobald `self.train` falsch ist."""
    features = state_to_features(game_state)

    # `self.epsilon` existiert nur im Training (siehe train.setup_training),
    # deshalb muss `self.train` zuerst geprüft werden.
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Exploring.")
        # 80 % laufen, 10 % warten, 10 % Bombe — Bomben sind selten sinnvoll,
        # also sollen sie auch selten ausprobiert werden.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    q_values = self.q_table[features]
    # Gleichstand zufällig brechen, sonst gewinnt bei der 0-initialisierten
    # Tabelle immer 'UP' und der Agent läuft in der ersten Runde gegen die Wand.
    best = np.flatnonzero(q_values == q_values.max())
    action = ACTIONS[np.random.choice(best)]
    self.logger.debug(f"State {features} -> {action}")
    return action

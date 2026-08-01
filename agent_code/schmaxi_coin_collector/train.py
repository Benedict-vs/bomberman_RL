"""Trainingsteil von Modell A. Nur mit `--train N` geladen, nie im Turnier."""

import pickle
from typing import List

import events as e
from .callbacks import MODEL_FILE
from .features import ACTIONS, state_to_features

try:
    from tools.trainlog import TrainLogger
except ImportError:          # tools/ ist nicht Teil der Abgabe
    TrainLogger = None


# --- Hyperparameter -------------------------------------------------------
# Startwerte aus KONZEPT §4, keine Wahrheit. Sie gehören in die systematische
# Optimierung — Hyperparametersuche ist ein eigener Bewertungspunkt.
ALPHA = 0.1          # Lernrate
GAMMA = 0.95         # Diskontierung
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995   # pro Episode; ~0.05 nach etwa 6000 Episoden

# ----CHANGE EACH RUN TO A NEW NAME, OTHERWISE TRAINING DATA WILL BE APPENDED TO
RUN_NAME = "q_v1_task1"
# ----CHANGE EACH RUN TO A NEW NAME, OTHERWISE TRAINING DATA WILL BE APPENDED TO


def setup_training(self):
    """Wird nach `setup` aus callbacks.py aufgerufen, vor dem ersten `act`."""
    self.epsilon = EPS_START

    self.trainlog = TrainLogger(
        agent="schmaxi_coin_collector",
        run=RUN_NAME,
        hyperparams={"alpha": ALPHA, "gamma": GAMMA, "eps_start": EPS_START,
                     "eps_end": EPS_END, "eps_decay": EPS_DECAY},
        # Ohne diese Zeile verwirft der DictWriter das `extra`-Feld unten
        # kommentarlos (extrasaction="ignore").
        extra_columns=["states"],
    ) if TrainLogger else None

    self.episode_events = []     # sammelt die Ereignisse der laufenden Episode
    self.episode_reward = 0.0


def update_q(self, old_features, action, reward, new_features):
    """Ein Q-Learning-Schritt.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

    `new_features is None` markiert einen Endzustand — dort gibt es nichts mehr
    zu bootstrappen, sonst lernt der Agent einen Wert für ein Leben nach dem Tod.
    """
    if old_features is None or action is None:
        return

    index = ACTIONS.index(action)
    bootstrap = 0.0 if new_features is None else GAMMA * self.q_table[new_features].max()
    old_value = self.q_table[old_features][index]
    self.q_table[old_features][index] = old_value + ALPHA * (reward + bootstrap - old_value)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """Ein Schritt: Belohnung bilden, Q-Tabelle aktualisieren, Episode mitschreiben."""
    self.logger.debug(f'Events {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    update_q(self, old_features, self_action, reward, new_features)

    self.episode_events.extend(events)
    self.episode_reward += reward


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Letzter Schritt der Episode: Endzustands-Update, Modell sichern, loggen."""
    self.logger.debug(f'Events {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    update_q(self, state_to_features(last_game_state), last_action, reward, None)

    self.episode_events.extend(events)
    self.episode_reward += reward

    self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    # `dict(...)` statt des defaultdict: der geladene Agent baut sich seinen
    # eigenen default_factory, und ein gepickeltes defaultdict schleppt die
    # Fabrik unnötig mit.
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(dict(self.q_table), file)

    if self.trainlog:
        self.trainlog.log_episode(
            episode=last_game_state["round"],
            score=last_game_state["self"][1],
            steps=last_game_state["step"],
            events=self.episode_events,
            reward=self.episode_reward,
            epsilon=self.epsilon,
            extra={"states": len(self.q_table)},
        )

    self.episode_events = []
    self.episode_reward = 0.0


# --- Belohnungen ----------------------------------------------------------
# Werte aus KONZEPT §4. Für Stufe 1 tragen nur COIN_COLLECTED, INVALID_ACTION,
# WAITED und die Schrittstrafe; der Rest steht schon hier, damit Stufe 2 keine
# Änderung an dieser Tabelle braucht und die Läufe vergleichbar bleiben.
STEP_PENALTY = -0.2

GAME_REWARDS = {
    e.COIN_COLLECTED: 5,
    e.COIN_FOUND: 2,
    e.CRATE_DESTROYED: 1,
    e.KILLED_OPPONENT: 25,
    e.KILLED_SELF: -30,
    e.GOT_KILLED: -15,
    e.INVALID_ACTION: -3,
    e.WAITED: -0.5,
}


def reward_from_events(self, events: List[str]) -> float:
    """Summe der Ereignisbelohnungen plus konstante Schrittstrafe."""
    reward_sum = STEP_PENALTY
    for event in events:
        reward_sum += GAME_REWARDS.get(event, 0)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

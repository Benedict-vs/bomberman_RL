"""Inference callbacks for the coin-heaven DQN agent."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .features import state_to_features
from .model import ACTIONS, CoinCollectorDQN


MODEL_FILE = "my-saved-model.pt"


TRAINING_SEED = 20260805
START_FROM_SAVED_MODEL = False


def setup(self) -> None:
    """Create the online network and optionally load trained parameters."""
    self.device = torch.device("cpu")

    if self.train:
        random.seed(TRAINING_SEED)
        np.random.seed(TRAINING_SEED)
        torch.manual_seed(TRAINING_SEED)

    self.online_network = CoinCollectorDQN().to(self.device)

    should_load_model = (
        os.path.isfile(MODEL_FILE)
        and (
            not self.train
            or START_FROM_SAVED_MODEL
        )
    )

    if should_load_model:
        state_dict = torch.load(
            MODEL_FILE,
            map_location=self.device,
            weights_only=True,
        )

        self.online_network.load_state_dict(state_dict)

        self.logger.info(
            "Loaded DQN parameters from %s.",
            MODEL_FILE,
        )

    elif self.train:
        self.logger.info(
            "Starting training from random parameters with seed %d.",
            TRAINING_SEED,
        )

    else:
        self.logger.warning(
            "No saved DQN found. Using random network parameters."
        )

    self.online_network.eval()


def act(self, game_state: dict) -> str:
    """Choose an action using epsilon-greedy exploration."""
    if self.train and random.random() < self.epsilon:
        action = random.choice(ACTIONS)

        self.logger.debug(
            "Exploration selected action %s at epsilon %.4f.",
            action,
            self.epsilon,
        )

        return action

    features = state_to_features(game_state)

    state_tensor = torch.from_numpy(features).unsqueeze(0)
    state_tensor = state_tensor.to(
        device=self.device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        q_values = self.online_network(state_tensor)

        action_index = int(
            q_values.argmax(dim=1).item()
        )

    action = ACTIONS[action_index]

    self.logger.debug(
        "Q-values %s, selected action %s.",
        q_values.squeeze(0).cpu().tolist(),
        action,
    )

    return action

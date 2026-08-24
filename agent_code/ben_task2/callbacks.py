"""Inference callbacks for the task-2 DQN agent."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .features import legal_action_mask, state_to_features
from .model import ACTIONS, CoinCollectorDQN


MODEL_FILE = "ben_task2_safety_potential_v1_15000ep_seed11.pt"


TRAINING_SEED = 11
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
    features = state_to_features(game_state)
    action_mask = legal_action_mask(features)
    legal_action_indices = np.flatnonzero(action_mask)

    if self.train and random.random() < self.epsilon:
        action_index = int(random.choice(legal_action_indices))
        action = ACTIONS[action_index]

        self.logger.debug(
            "Exploration selected action %s at epsilon %.4f.",
            action,
            self.epsilon,
        )

        return action

    state_tensor = torch.from_numpy(features).unsqueeze(0)
    state_tensor = state_tensor.to(
        device=self.device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        q_values = self.online_network(state_tensor)
        mask_tensor = torch.from_numpy(action_mask).to(self.device)
        masked_q_values = q_values.masked_fill(
            ~mask_tensor.unsqueeze(0),
            -torch.inf,
        )

        action_index = int(
            masked_q_values.argmax(dim=1).item()
        )

    action = ACTIONS[action_index]

    self.logger.debug(
        "Q-values %s, selected action %s.",
        q_values.squeeze(0).cpu().tolist(),
        action,
    )

    return action

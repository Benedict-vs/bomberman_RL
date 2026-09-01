"""Frozen inference callbacks for the fine-tuned task-2 DQN agent."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .features import legal_action_mask, state_to_features
from .model import ACTIONS, CoinCollectorDQN


MODEL_FILE = "dqn_task2_high_score_finetuned_seed11.pt"
LOAD_MODEL_FILE = MODEL_FILE
TRAINING_SEED = 11
START_FROM_SAVED_MODEL = False
VISIT_COUNT_ENABLED = True
VISIT_COUNT_ENCODING = "linear_10"
ESCAPE_FEATURE_MODE = "reachable_safe_tiles"


def setup(self) -> None:
    """Create the online network and optionally load trained parameters."""
    self.device = torch.device("cpu")

    if self.train:
        random.seed(TRAINING_SEED)
        np.random.seed(TRAINING_SEED)
        torch.manual_seed(TRAINING_SEED)

    self.online_network = CoinCollectorDQN(input_channels=10).to(self.device)

    model_to_load = (
        LOAD_MODEL_FILE
        if self.train and START_FROM_SAVED_MODEL
        else MODEL_FILE
    )
    should_load_model = (
        os.path.isfile(model_to_load)
        and (
            not self.train
            or START_FROM_SAVED_MODEL
        )
    )

    if should_load_model:
        state_dict = torch.load(
            model_to_load,
            map_location=self.device,
            weights_only=True,
        )

        self.online_network.load_state_dict(state_dict)

        self.logger.info(
            "Loaded DQN parameters from %s.",
            model_to_load,
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
    self.visit_round = None
    self.visit_counts = None
    self.last_action_features = None
    self.last_feature_round_step = None
    self.visit_count_encoding = VISIT_COUNT_ENCODING


def act(self, game_state: dict) -> str:
    """Choose an action using epsilon-greedy exploration."""
    features = _features_with_visit_count(self, game_state)
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


def _features_with_visit_count(self, game_state: dict) -> np.ndarray:
    """Count the current tile once and return the augmented features."""
    if not VISIT_COUNT_ENABLED:
        augmented_state = dict(game_state)
        augmented_state["escape_feature_mode"] = ESCAPE_FEATURE_MODE
        features = state_to_features(augmented_state)
        self.last_action_features = features.copy()
        self.last_feature_round_step = (
            game_state.get("round"),
            game_state.get("step"),
        )
        return features

    field = game_state["field"]
    round_number = game_state.get("round")

    if self.visit_round != round_number or self.visit_counts is None:
        self.visit_round = round_number
        self.visit_counts = np.zeros_like(field, dtype=np.float32)

    self_x, self_y = game_state["self"][3]
    self.visit_counts[self_x, self_y] += 1.0

    augmented_state = dict(game_state)
    augmented_state["visit_counts"] = self.visit_counts
    augmented_state["visit_count_encoding"] = self.visit_count_encoding
    augmented_state["escape_feature_mode"] = ESCAPE_FEATURE_MODE
    features = state_to_features(augmented_state)

    self.last_action_features = features.copy()
    self.last_feature_round_step = (
        game_state.get("round"),
        game_state.get("step"),
    )
    return features

"""Inference callbacks for the task-2 DQN agent."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .features import legal_action_mask, state_to_features
from .model import ACTIONS, CoinCollectorDQN


MULTISEED_ARM = os.environ.get("BM_TASK2_ESCAPE_ARM")
TRAINING_SEED = int(os.environ.get("BM_TASK2_TRAINING_SEED", "11"))
TOTAL_EPISODES = int(os.environ.get("BM_TASK2_TOTAL_EPISODES", "7000"))
FINE_TUNE_CRATE_WAIT = (
    os.environ.get("BM_TASK2_FINETUNE_CRATE_WAIT", "0") == "1"
)
FINE_TUNE_COIN_POTENTIAL = (
    os.environ.get("BM_TASK2_FINETUNE_COIN_POTENTIAL", "0") == "1"
)
FINE_TUNE_COIN_REWARD = (
    os.environ.get("BM_TASK2_FINETUNE_COIN_REWARD", "0") == "1"
)
CONTINUE_COIN_REWARD15 = (
    os.environ.get("BM_TASK2_CONTINUE_COIN_REWARD15", "0") == "1"
)
CRATE_WAIT_PENALTY = float(
    os.environ.get("BM_TASK2_CRATE_WAIT_PENALTY", "-0.02")
)
if TOTAL_EPISODES <= 0:
    raise ValueError("BM_TASK2_TOTAL_EPISODES must be positive.")
if FINE_TUNE_COIN_POTENTIAL and FINE_TUNE_COIN_REWARD:
    raise ValueError(
        "Coin-potential and coin-reward fine-tuning are separate arms."
    )
if CONTINUE_COIN_REWARD15:
    if (
        not FINE_TUNE_CRATE_WAIT
        or not FINE_TUNE_COIN_REWARD
        or FINE_TUNE_COIN_POTENTIAL
        or MULTISEED_ARM != "reachable"
        or TOTAL_EPISODES != 5000
        or CRATE_WAIT_PENALTY != -0.03
    ):
        raise ValueError(
            "Coin-reward continuation requires the reachable "
            "5000-episode reward-1.5 crate-WAIT -0.03 configuration."
        )
    EXPERIMENT_STEM = (
        "escape_crate_wait003_coin_reward15_"
        "continue5000_from_reward15_v1"
    )
    ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
elif FINE_TUNE_COIN_REWARD:
    if (
        not FINE_TUNE_CRATE_WAIT
        or MULTISEED_ARM != "reachable"
        or TOTAL_EPISODES != 2000
        or CRATE_WAIT_PENALTY != -0.03
    ):
        raise ValueError(
            "Coin-reward fine-tuning requires the reachable "
            "2000-episode crate-WAIT -0.03 configuration."
        )
    EXPERIMENT_STEM = (
        "escape_crate_wait003_coin_reward15_"
        "finetune2000_from_wait003_v1"
    )
    ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
elif FINE_TUNE_COIN_POTENTIAL:
    if (
        not FINE_TUNE_CRATE_WAIT
        or MULTISEED_ARM != "reachable"
        or TOTAL_EPISODES != 2000
        or CRATE_WAIT_PENALTY != -0.03
    ):
        raise ValueError(
            "Coin-potential fine-tuning requires the reachable "
            "2000-episode crate-WAIT -0.03 configuration."
        )
    EXPERIMENT_STEM = (
        "escape_crate_wait003_coin_potential_"
        "finetune2000_from_wait003_v1"
    )
    ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
elif FINE_TUNE_CRATE_WAIT:
    if MULTISEED_ARM != "reachable" or TOTAL_EPISODES != 2000:
        raise ValueError(
            "Crate-WAIT fine-tuning requires reachable and 2000 episodes."
        )
    if CRATE_WAIT_PENALTY not in (-0.02, -0.03):
        raise ValueError("Crate-WAIT penalty must be -0.02 or -0.03.")
    penalty_name = "002" if CRATE_WAIT_PENALTY == -0.02 else "003"
    EXPERIMENT_STEM = (
        f"escape_crate_wait{penalty_name}_finetune2000_from10000_v1"
    )
    ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
elif MULTISEED_ARM is None:
    EXPERIMENT_STEM = "escape_reachable_tiles_v1"
    ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
elif MULTISEED_ARM == "zero":
    EXPERIMENT_STEM = "escape_multiseed_zero_v1"
    ESCAPE_FEATURE_MODE = "zero"
elif MULTISEED_ARM == "reachable":
    EXPERIMENT_STEM = "escape_multiseed_reachable_v1"
    ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
else:
    raise ValueError(
        "BM_TASK2_ESCAPE_ARM must be 'zero' or 'reachable'."
    )

if CONTINUE_COIN_REWARD15:
    MODEL_FILE = f"ben_task2_{EXPERIMENT_STEM}_seed{TRAINING_SEED}.pt"
    LOAD_MODEL_FILE = (
        "ben_task2_escape_crate_wait003_coin_reward15_"
        "finetune2000_from_wait003_v1_seed"
        f"{TRAINING_SEED}.pt"
    )
elif FINE_TUNE_COIN_REWARD:
    MODEL_FILE = f"ben_task2_{EXPERIMENT_STEM}_seed{TRAINING_SEED}.pt"
    LOAD_MODEL_FILE = (
        "ben_task2_escape_crate_wait003_"
        "finetune2000_from10000_v1_seed"
        f"{TRAINING_SEED}.pt"
    )
elif FINE_TUNE_COIN_POTENTIAL:
    MODEL_FILE = f"ben_task2_{EXPERIMENT_STEM}_seed{TRAINING_SEED}.pt"
    LOAD_MODEL_FILE = (
        "ben_task2_escape_crate_wait003_"
        "finetune2000_from10000_v1_seed"
        f"{TRAINING_SEED}.pt"
    )
elif FINE_TUNE_CRATE_WAIT:
    MODEL_FILE = f"ben_task2_{EXPERIMENT_STEM}_seed{TRAINING_SEED}.pt"
    LOAD_MODEL_FILE = (
        "ben_task2_escape_multiseed_reachable_v1_10000ep_seed"
        f"{TRAINING_SEED}.pt"
    )
else:
    MODEL_FILE = (
        f"ben_task2_{EXPERIMENT_STEM}_{TOTAL_EPISODES}ep_seed"
        f"{TRAINING_SEED}.pt"
    )
    LOAD_MODEL_FILE = MODEL_FILE


START_FROM_SAVED_MODEL = (
    FINE_TUNE_CRATE_WAIT
    or FINE_TUNE_COIN_POTENTIAL
    or FINE_TUNE_COIN_REWARD
    or CONTINUE_COIN_REWARD15
)
VISIT_COUNT_ENABLED = True
VISIT_COUNT_ENCODING = "linear_10"


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

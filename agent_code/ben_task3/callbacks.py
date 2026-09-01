"""Inference callbacks for Ben's task-3 DQN development agent."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .features import legal_action_mask, state_to_features
from .model import ACTIONS, CoinCollectorDQN


TRAINING_SEED = int(os.environ.get("BM_TASK3_TRAINING_SEED", "11"))
TOTAL_EPISODES = int(os.environ.get("BM_TASK3_TOTAL_EPISODES", "5000"))
if TOTAL_EPISODES <= 0:
    raise ValueError("BM_TASK3_TOTAL_EPISODES must be positive.")

# Task 3 starts from the frozen, audited task-2 policy. The first task-3
# experiment is named independently and never writes into ben_task2.
MULTISEED_ARM = "reachable"
FINE_TUNE_CRATE_WAIT = True
FINE_TUNE_COIN_POTENTIAL = False
FINE_TUNE_COIN_REWARD = True
CONTINUE_COIN_REWARD15 = False
CRATE_WAIT_PENALTY = -0.03
TRAINING_ARM = os.environ.get("BM_TASK3_TRAINING_ARM", "peaceful_v1")
if TRAINING_ARM == "peaceful_v1":
    EXPERIMENT_STEM = "peaceful_v1"
    LOAD_MODEL_FILE = "ben_task3_task2_baseline_11ch_seed11.pt"
elif TRAINING_ARM == "coin_collector_finetune_v1":
    EXPERIMENT_STEM = "coin_collector_finetune_v1"
    LOAD_MODEL_FILE = "ben_task3_peaceful_v1_5000ep_seed11.pt"
elif TRAINING_ARM == "coin_collector_kill_reward75_v1":
    EXPERIMENT_STEM = "coin_collector_kill_reward75_v1"
    LOAD_MODEL_FILE = (
        "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt"
    )
elif TRAINING_ARM == "coin_collector_opponent_potential_v1":
    EXPERIMENT_STEM = "coin_collector_opponent_potential_v1"
    LOAD_MODEL_FILE = (
        "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt"
    )
elif TRAINING_ARM == "mixed_curriculum_v1":
    EXPERIMENT_STEM = "mixed_curriculum_v1"
    LOAD_MODEL_FILE = (
        "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt"
    )
elif TRAINING_ARM == "mixed_safety_multiseed_v1":
    EXPERIMENT_STEM = "mixed_safety_multiseed_v1"
    LOAD_MODEL_FILE = "ben_task3_mixed_curriculum_v1_2000ep_seed11.pt"
else:
    raise ValueError(
        "Unknown BM_TASK3_TRAINING_ARM."
    )
ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
MODEL_FILE = (
    f"ben_task3_{EXPERIMENT_STEM}_{TOTAL_EPISODES}ep_seed"
    f"{TRAINING_SEED}.pt"
)
START_FROM_SAVED_MODEL = True
_AGENT_DIRECTORY = os.path.dirname(__file__)
MODEL_VARIANT = os.environ.get("BM_TASK3_MODEL_VARIANT", "auto")
if MODEL_VARIANT == "baseline":
    INFERENCE_MODEL_FILE = LOAD_MODEL_FILE
elif MODEL_VARIANT == "trained":
    if not os.path.isfile(os.path.join(_AGENT_DIRECTORY, MODEL_FILE)):
        raise FileNotFoundError(f"Task-3 trained model not found: {MODEL_FILE}")
    INFERENCE_MODEL_FILE = MODEL_FILE
elif MODEL_VARIANT == "auto":
    INFERENCE_MODEL_FILE = (
        MODEL_FILE
        if os.path.isfile(os.path.join(_AGENT_DIRECTORY, MODEL_FILE))
        else LOAD_MODEL_FILE
    )
else:
    raise ValueError(
        "BM_TASK3_MODEL_VARIANT must be 'auto', 'baseline', or 'trained'."
    )
CHECKPOINT_EPISODE_TEXT = os.environ.get("BM_TASK3_CHECKPOINT_EPISODE")
CHECKPOINT_EPISODE = (
    int(CHECKPOINT_EPISODE_TEXT)
    if CHECKPOINT_EPISODE_TEXT is not None
    else None
)
if CHECKPOINT_EPISODE is not None:
    if (
        MODEL_VARIANT != "trained"
        or CHECKPOINT_EPISODE <= 0
        or CHECKPOINT_EPISODE > TOTAL_EPISODES
        or CHECKPOINT_EPISODE % 100 != 0
    ):
        raise ValueError(
            "BM_TASK3_CHECKPOINT_EPISODE requires MODEL_VARIANT=trained "
            "and a 100-step checkpoint within TOTAL_EPISODES."
        )
    checkpoint_name = (
        f"task3_{EXPERIMENT_STEM}_{TOTAL_EPISODES}ep_seed"
        f"{TRAINING_SEED}__episode_{CHECKPOINT_EPISODE}.pt"
    )
    INFERENCE_MODEL_FILE = f"../../results/train/ben_task3/{checkpoint_name}"
    if not os.path.isfile(
        os.path.join(_AGENT_DIRECTORY, INFERENCE_MODEL_FILE)
    ):
        raise FileNotFoundError(
            f"Task-3 checkpoint not found: {INFERENCE_MODEL_FILE}"
        )
VISIT_COUNT_ENABLED = True
VISIT_COUNT_ENCODING = "linear_10"


def setup(self) -> None:
    """Create the online network and optionally load trained parameters."""
    self.device = torch.device("cpu")

    if self.train and CHECKPOINT_EPISODE is not None:
        raise ValueError("BM_TASK3_CHECKPOINT_EPISODE is evaluation-only.")

    if self.train:
        random.seed(TRAINING_SEED)
        np.random.seed(TRAINING_SEED)
        torch.manual_seed(TRAINING_SEED)

    self.online_network = CoinCollectorDQN(input_channels=11).to(self.device)

    if self.train and START_FROM_SAVED_MODEL:
        model_to_load = LOAD_MODEL_FILE
    else:
        model_to_load = INFERENCE_MODEL_FILE
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

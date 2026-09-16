"""Inference callbacks for Ben's isolated Task-4 DQN development agent."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .features import legal_action_mask, state_to_features
from .model import ACTIONS, BombOutcomeDQN, CoinCollectorDQN, DuelingCoinCollectorDQN, OpponentPredictionDQN


TRAINING_SEED = int(os.environ.get("BM_TASK4_TRAINING_SEED", "11"))
TOTAL_EPISODES = int(os.environ.get("BM_TASK4_TOTAL_EPISODES", "5000"))
if TOTAL_EPISODES <= 0:
    raise ValueError("BM_TASK4_TOTAL_EPISODES must be positive.")

# Task 4 starts from the frozen, audited Task-3 seed-13 policy. Development
# artifacts use independent Task-4 names and never write into ben_task3.
MULTISEED_ARM = "reachable"
FINE_TUNE_CRATE_WAIT = True
FINE_TUNE_COIN_POTENTIAL = False
FINE_TUNE_COIN_REWARD = True
CONTINUE_COIN_REWARD15 = False
CRATE_WAIT_PENALTY = -0.03
TRAINING_ARM = os.environ.get(
    "BM_TASK4_TRAINING_ARM",
    "mixed_kill_v1",
)
ENSEMBLE_SECOND_MODEL_FILE = None
if TRAINING_ARM == "baseline_v1":
    EXPERIMENT_STEM = "baseline_v1"
    LOAD_MODEL_FILE = "ben_task4_task3_baseline_seed13.pt"
elif TRAINING_ARM == "rule_based_suicide7_v1":
    EXPERIMENT_STEM = "rule_based_suicide7_v1"
    LOAD_MODEL_FILE = "ben_task4_task3_baseline_seed13.pt"
elif TRAINING_ARM == "rule_based_continue_v1":
    EXPERIMENT_STEM = "rule_based_continue_v1"
    LOAD_MODEL_FILE = "ben_task4_baseline_v1_5000ep_seed11.pt"
elif TRAINING_ARM in {
    "bomb_outcome_control_v1",
    "bomb_outcome_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_bomb_outcome_mixed_kill_source.pt"
elif TRAINING_ARM in {
    "mixed_kill_v1",
    "mixed_kill_seed12_v1",
    "mixed_kill_seed13_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_baseline_v1_5000ep_seed11.pt"
elif TRAINING_ARM == "rule_based_continue_control1000_v1":
    EXPERIMENT_STEM = "rule_based_continue_control1000_v1"
    LOAD_MODEL_FILE = "ben_task4_rule_based_continue_v1_2000ep_seed11.pt"
elif TRAINING_ARM == "mixed_kill_consolidate1000_v1":
    EXPERIMENT_STEM = "mixed_kill_consolidate1000_v1"
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
elif TRAINING_ARM in {
    "terminal_safety_control_v1",
    "terminal_safety_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
elif TRAINING_ARM == "mixed_kill_continue7000_v1":
    EXPERIMENT_STEM = "mixed_kill_continue7000_v1"
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
elif TRAINING_ARM in {
    "opponent_bomb_tradeoff_zero_v1",
    "opponent_bomb_tradeoff_v1",
    "spacetime_safety_zero_v1",
    "spacetime_safety_v1",
    "coin_navigation_zero_v1",
    "coin_navigation_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_12ch_source.pt"
elif TRAINING_ARM in {
    "opponent_alignment_zero_v1",
    "opponent_alignment_v1",
    "opponent_alignment_mixed_zero_v1",
    "opponent_alignment_mixed_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_safe_seed11_12ch_alignment_source.pt"
elif TRAINING_ARM in {
    "safe_offense_zero_v1",
    "safe_offense_reward01_v1",
    "safe_offense_reward02_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = (
        "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
    )
elif TRAINING_ARM in {
    "opponent_escape_pressure_zero_v1",
    "opponent_escape_pressure_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_safe_seed11_12ch_alignment_source.pt"
elif TRAINING_ARM in {
    "external_trio_control_v1",
    "external_trio_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = (
        "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
    )
elif TRAINING_ARM in {
    "mixed_external_control_v1",
    "mixed_external_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
elif TRAINING_ARM == "ensemble_v1":
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
    ENSEMBLE_SECOND_MODEL_FILE = (
        "ben_task4_curriculum_candidate_p4_v1_250ep_seed11.pt"
    )
elif TRAINING_ARM in {
    "opponent_prediction_control_v1",
    "opponent_prediction_v1",
    "dueling_control_v1",
    "dueling_v1",
    "per_control_v1",
    "per_v1",
    "double_dqn_control_v1",
    "double_dqn_v1",
    "learning_rate_control_v1",
    "learning_rate5e5_v1",
    "event_replay_control_v1",
    "event_replay_balanced_v1",
    "kill_reward_control_p1_v1",
    "kill_reward_control_p2_v1",
    "kill_reward_control_p3_v1",
    "kill_reward_control_p4_v1",
    "kill_reward75_p1_v1",
    "kill_reward75_p2_v1",
    "kill_reward75_p3_v1",
    "kill_reward75_p4_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = (
        "ben_task4_opponent_prediction_mixed_kill_source.pt"
        if TRAINING_ARM == "opponent_prediction_v1"
        else "ben_task4_dueling_mixed_kill_source.pt"
        if TRAINING_ARM == "dueling_v1"
        else "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
    )
elif TRAINING_ARM in {
    "survival_penalty_control_v1",
    "survival_penalty10_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
elif TRAINING_ARM in {
    "curriculum_control_p1_v1",
    "curriculum_control_p2_v1",
    "curriculum_control_p3_v1",
    "curriculum_control_p4_v1",
    "curriculum_candidate_p1_v1",
    "curriculum_candidate_p2_v1",
    "curriculum_candidate_p3_v1",
    "curriculum_candidate_p4_v1",
    "opponent_rotation_control_p1_v1",
    "opponent_rotation_control_p2_v1",
    "opponent_rotation_control_p3_v1",
    "opponent_rotation_control_p4_v1",
    "opponent_rotation_candidate_p1_v1",
    "opponent_rotation_candidate_p2_v1",
    "opponent_rotation_candidate_p3_v1",
    "opponent_rotation_candidate_p4_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
elif TRAINING_ARM in {
    "opponent_bomb_ready_zero_v1",
    "opponent_bomb_ready_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_safe_seed11_12ch_alignment_source.pt"
elif TRAINING_ARM in {
    "exploration_eps05_v1",
    "exploration_eps15_v1",
    "exploration_eps05_long3000_v1",
    "exploration_eps15_long3000_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = (
        "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
    )
elif TRAINING_ARM in {
    "temporal_safety_zero_v1",
    "temporal_safety_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_12ch_source.pt"
elif TRAINING_ARM in {
    "nstep1_control_v1",
    "nstep3_v1",
    "full_symmetry_control_v1",
    "full_symmetry_v1",
}:
    EXPERIMENT_STEM = TRAINING_ARM
    LOAD_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
else:
    raise ValueError("Unknown BM_TASK4_TRAINING_ARM.")
LOAD_MODEL_OVERRIDE = os.environ.get("BM_TASK4_LOAD_MODEL_FILE")
if LOAD_MODEL_OVERRIDE:
    if os.path.isabs(LOAD_MODEL_OVERRIDE):
        raise ValueError("BM_TASK4_LOAD_MODEL_FILE must be relative.")
    LOAD_MODEL_FILE = LOAD_MODEL_OVERRIDE
ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
MODEL_FILE = (
    f"ben_task4_{EXPERIMENT_STEM}_{TOTAL_EPISODES}ep_seed"
    f"{TRAINING_SEED}.pt"
)
START_FROM_SAVED_MODEL = True
_AGENT_DIRECTORY = os.path.dirname(__file__)
# Inference deliberately names the selected policy independently of the next
# training artifact. A future output path must never silently select a policy.
INCUMBENT_INFERENCE_MODEL_FILE = "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
MODEL_VARIANT = os.environ.get("BM_TASK4_MODEL_VARIANT", "incumbent")
if MODEL_VARIANT == "incumbent":
    INFERENCE_MODEL_FILE = INCUMBENT_INFERENCE_MODEL_FILE
elif MODEL_VARIANT == "baseline":
    INFERENCE_MODEL_FILE = LOAD_MODEL_FILE
elif MODEL_VARIANT == "trained":
    if not os.path.isfile(os.path.join(_AGENT_DIRECTORY, MODEL_FILE)):
        raise FileNotFoundError(f"Task-4 trained model not found: {MODEL_FILE}")
    INFERENCE_MODEL_FILE = MODEL_FILE
else:
    raise ValueError(
        "BM_TASK4_MODEL_VARIANT must be 'incumbent', 'baseline', or 'trained'."
    )
if not os.path.isfile(os.path.join(_AGENT_DIRECTORY, INFERENCE_MODEL_FILE)):
    raise FileNotFoundError(
        f"Task-4 inference model not found: {INFERENCE_MODEL_FILE}"
    )
CHECKPOINT_EPISODE_TEXT = os.environ.get("BM_TASK4_CHECKPOINT_EPISODE")
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
            "BM_TASK4_CHECKPOINT_EPISODE requires MODEL_VARIANT=trained "
            "and a 100-step checkpoint within TOTAL_EPISODES."
        )
    checkpoint_name = (
        f"task4_{EXPERIMENT_STEM}_{TOTAL_EPISODES}ep_seed"
        f"{TRAINING_SEED}__episode_{CHECKPOINT_EPISODE}.pt"
    )
    INFERENCE_MODEL_FILE = f"../../results/train/ben_task4/{checkpoint_name}"
    if not os.path.isfile(
        os.path.join(_AGENT_DIRECTORY, INFERENCE_MODEL_FILE)
    ):
        raise FileNotFoundError(
            f"Task-4 checkpoint not found: {INFERENCE_MODEL_FILE}"
        )
VISIT_COUNT_ENABLED = True
VISIT_COUNT_ENCODING = "linear_10"
OPPONENT_ALIGNMENT_MODE = (
    "enabled"
    if TRAINING_ARM
    in {"opponent_alignment_v1", "opponent_alignment_mixed_v1"}
    else "zero"
    if TRAINING_ARM
    in {"opponent_alignment_zero_v1", "opponent_alignment_mixed_zero_v1"}
    else "disabled"
)
OPPONENT_ESCAPE_PRESSURE_MODE = (
    "enabled"
    if TRAINING_ARM == "opponent_escape_pressure_v1"
    else "zero"
    if TRAINING_ARM == "opponent_escape_pressure_zero_v1"
    else "disabled"
)
OPPONENT_BOMB_READY_MODE = (
    "enabled"
    if TRAINING_ARM == "opponent_bomb_ready_v1"
    else "zero"
    if TRAINING_ARM == "opponent_bomb_ready_zero_v1"
    else "disabled"
)
TEMPORAL_SAFETY_MODE = (
    "enabled"
    if TRAINING_ARM == "temporal_safety_v1"
    else "zero"
    if TRAINING_ARM == "temporal_safety_zero_v1"
    else "disabled"
)
OPPONENT_BOMB_TRADEOFF_MODE = (
    "enabled"
    if TRAINING_ARM == "opponent_bomb_tradeoff_v1"
    else "zero"
    if TRAINING_ARM == "opponent_bomb_tradeoff_zero_v1"
    else "disabled"
)
SPACETIME_SAFETY_MODE = (
    "enabled"
    if TRAINING_ARM == "spacetime_safety_v1"
    else "zero"
    if TRAINING_ARM == "spacetime_safety_zero_v1"
    else "disabled"
)
COIN_NAVIGATION_MODE = (
    "enabled" if TRAINING_ARM == "coin_navigation_v1" else "zero"
    if TRAINING_ARM == "coin_navigation_zero_v1" else "disabled"
)
INPUT_CHANNELS = (
    12
    if OPPONENT_ALIGNMENT_MODE != "disabled"
    or OPPONENT_ESCAPE_PRESSURE_MODE != "disabled"
    or OPPONENT_BOMB_READY_MODE != "disabled"
    or TEMPORAL_SAFETY_MODE != "disabled"
    or OPPONENT_BOMB_TRADEOFF_MODE != "disabled"
    or SPACETIME_SAFETY_MODE != "disabled"
    or COIN_NAVIGATION_MODE != "disabled"
    else 11
)
DUELING_DQN = TRAINING_ARM == "dueling_v1"
OPPONENT_PREDICTION = TRAINING_ARM == "opponent_prediction_v1"
BOMB_OUTCOME_AUXILIARY = TRAINING_ARM == "bomb_outcome_v1"
FULL_SYMMETRY_AUGMENTATION = TRAINING_ARM == "full_symmetry_v1"
NETWORK_CLASS = (
    BombOutcomeDQN if TRAINING_ARM in {"bomb_outcome_control_v1", "bomb_outcome_v1"} else OpponentPredictionDQN if OPPONENT_PREDICTION else DuelingCoinCollectorDQN
    if DUELING_DQN else CoinCollectorDQN
)


def setup(self) -> None:
    """Create the online network and optionally load trained parameters."""
    self.device = torch.device("cpu")

    if self.train and CHECKPOINT_EPISODE is not None:
        raise ValueError("BM_TASK4_CHECKPOINT_EPISODE is evaluation-only.")

    if self.train:
        random.seed(TRAINING_SEED)
        np.random.seed(TRAINING_SEED)
        torch.manual_seed(TRAINING_SEED)

    self.online_network = NETWORK_CLASS(input_channels=INPUT_CHANNELS).to(
        self.device
    )

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

        if ENSEMBLE_SECOND_MODEL_FILE is not None:
            self.ensemble_network = CoinCollectorDQN(
                input_channels=INPUT_CHANNELS
            ).to(self.device)
            second_path = os.path.join(
                _AGENT_DIRECTORY,
                ENSEMBLE_SECOND_MODEL_FILE,
            )
            second_state_dict = torch.load(
                second_path,
                map_location=self.device,
                weights_only=True,
            )
            self.ensemble_network.load_state_dict(second_state_dict)
            self.ensemble_network.eval()
            self.logger.info(
                "Loaded ensemble parameters from %s.",
                ENSEMBLE_SECOND_MODEL_FILE,
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
        if ENSEMBLE_SECOND_MODEL_FILE is not None:
            q_values = (
                q_values + self.ensemble_network(state_tensor)
            ) / 2.0
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
        augmented_state["opponent_alignment_mode"] = OPPONENT_ALIGNMENT_MODE
        augmented_state["opponent_escape_pressure_mode"] = (
            OPPONENT_ESCAPE_PRESSURE_MODE
        )
        augmented_state["opponent_bomb_ready_mode"] = (
            OPPONENT_BOMB_READY_MODE
        )
        augmented_state["temporal_safety_mode"] = TEMPORAL_SAFETY_MODE
        augmented_state["opponent_bomb_tradeoff_mode"] = (
            OPPONENT_BOMB_TRADEOFF_MODE
        )
        augmented_state["spacetime_safety_mode"] = SPACETIME_SAFETY_MODE
        augmented_state["coin_navigation_mode"] = COIN_NAVIGATION_MODE
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
    augmented_state["opponent_alignment_mode"] = OPPONENT_ALIGNMENT_MODE
    augmented_state["opponent_escape_pressure_mode"] = (
        OPPONENT_ESCAPE_PRESSURE_MODE
    )
    augmented_state["opponent_bomb_ready_mode"] = OPPONENT_BOMB_READY_MODE
    augmented_state["temporal_safety_mode"] = TEMPORAL_SAFETY_MODE
    augmented_state["opponent_bomb_tradeoff_mode"] = (
        OPPONENT_BOMB_TRADEOFF_MODE
    )
    augmented_state["spacetime_safety_mode"] = SPACETIME_SAFETY_MODE
    augmented_state["coin_navigation_mode"] = COIN_NAVIGATION_MODE
    features = state_to_features(augmented_state)

    self.last_action_features = features.copy()
    self.last_feature_round_step = (
        game_state.get("round"),
        game_state.get("step"),
    )
    return features

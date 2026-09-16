"""Fixed CPU inference callbacks for the final Agent B submission."""

from __future__ import annotations

import numpy as np
import torch

from .features import legal_action_mask, state_to_features
from .model import ACTIONS, CoinCollectorDQN


MODEL_FILE = "Agent_B_model.pt"
ESCAPE_FEATURE_MODE = "reachable_safe_tiles"
VISIT_COUNT_ENABLED = True
VISIT_COUNT_ENCODING = "linear_10"
OPPONENT_ALIGNMENT_MODE = "disabled"
OPPONENT_ESCAPE_PRESSURE_MODE = "disabled"
OPPONENT_BOMB_READY_MODE = "disabled"
TEMPORAL_SAFETY_MODE = "disabled"
INPUT_CHANNELS = 11


def setup(self) -> None:
    """Load the fixed trained policy for CPU-only tournament inference."""
    self.device = torch.device("cpu")
    self.online_network = CoinCollectorDQN(input_channels=INPUT_CHANNELS).to(
        self.device
    )
    state_dict = torch.load(
        MODEL_FILE,
        map_location=self.device,
        weights_only=True,
    )
    self.online_network.load_state_dict(state_dict)
    self.logger.info("Loaded fixed Agent B parameters from %s.", MODEL_FILE)

    self.online_network.eval()
    self.visit_round = None
    self.visit_counts = None
    self.last_action_features = None
    self.last_feature_round_step = None
    self.visit_count_encoding = VISIT_COUNT_ENCODING


def act(self, game_state: dict) -> str:
    """Choose the greedy legal action of the fixed learned policy."""
    features = _features_with_visit_count(self, game_state)
    action_mask = legal_action_mask(features)

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
        augmented_state["opponent_alignment_mode"] = OPPONENT_ALIGNMENT_MODE
        augmented_state["opponent_escape_pressure_mode"] = (
            OPPONENT_ESCAPE_PRESSURE_MODE
        )
        augmented_state["opponent_bomb_ready_mode"] = (
            OPPONENT_BOMB_READY_MODE
        )
        augmented_state["temporal_safety_mode"] = TEMPORAL_SAFETY_MODE
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
    features = state_to_features(augmented_state)

    self.last_action_features = features.copy()
    self.last_feature_round_step = (
        game_state.get("round"),
        game_state.get("step"),
    )
    return features

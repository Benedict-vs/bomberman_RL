"""Evaluation-only wrapper for the trained task-2 visit-count policy."""

from __future__ import annotations

import os

import numpy as np
import torch

from agent_code.ben_task2.features import legal_action_mask, state_to_features
from agent_code.ben_task2.model import ACTIONS, CoinCollectorDQN


MODEL_FILE = "ben_task2_visit_count_v1_7000ep_seed11.pt"


def setup(self) -> None:
    """Load the frozen visit-count model for greedy CPU evaluation."""
    self.device = torch.device("cpu")
    self.online_network = CoinCollectorDQN().to(self.device)

    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"Missing visit-count model: {MODEL_FILE}")

    state_dict = torch.load(
        MODEL_FILE,
        map_location=self.device,
        weights_only=True,
    )
    self.online_network.load_state_dict(state_dict)
    self.online_network.eval()
    self.visit_round = None
    self.visit_counts = None


def act(self, game_state: dict) -> str:
    """Choose greedily while maintaining the trained visit-count channel."""
    features = _features_with_visit_count(self, game_state)
    action_mask = legal_action_mask(features)
    state_tensor = torch.from_numpy(features).unsqueeze(0).float()

    with torch.inference_mode():
        q_values = self.online_network(state_tensor)[0]
        mask = torch.from_numpy(action_mask)
        q_values = q_values.masked_fill(~mask, -torch.inf)

    return ACTIONS[int(q_values.argmax().item())]


def _features_with_visit_count(self, game_state: dict) -> np.ndarray:
    """Count the current tile once and add the normalized ninth channel."""
    field = game_state["field"]
    round_number = game_state.get("round")

    if self.visit_round != round_number or self.visit_counts is None:
        self.visit_round = round_number
        self.visit_counts = np.zeros_like(field, dtype=np.float32)

    self_x, self_y = game_state["self"][3]
    self.visit_counts[self_x, self_y] += 1.0

    augmented_state = dict(game_state)
    augmented_state["visit_counts"] = self.visit_counts
    augmented_state["visit_count_encoding"] = "linear_10"
    return state_to_features(augmented_state)

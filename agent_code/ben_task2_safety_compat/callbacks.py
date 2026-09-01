"""Evaluation-only wrapper for the historical 8-channel safety policy."""

from __future__ import annotations

import os

import numpy as np
import torch

from agent_code.ben_task2.features import legal_action_mask, state_to_features
from agent_code.ben_task2.model import ACTIONS, CoinCollectorDQN


MODEL_FILE = "ben_task2_safety_compat_9ch_7000ep_seed11.pt"


def setup(self) -> None:
    """Load the zero-extended safety model for CPU-only evaluation."""
    self.device = torch.device("cpu")
    self.online_network = CoinCollectorDQN().to(self.device)

    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"Missing converted safety model: {MODEL_FILE}")

    state_dict = torch.load(
        MODEL_FILE,
        map_location=self.device,
        weights_only=True,
    )
    self.online_network.load_state_dict(state_dict)
    self.online_network.eval()


def act(self, game_state: dict) -> str:
    """Choose greedily; the ninth input channel remains exactly zero."""
    features = state_to_features(game_state)
    action_mask = legal_action_mask(features)
    state_tensor = torch.from_numpy(features).unsqueeze(0).float()

    with torch.inference_mode():
        q_values = self.online_network(state_tensor)[0]
        mask = torch.from_numpy(action_mask)
        q_values = q_values.masked_fill(~mask, -torch.inf)

    return ACTIONS[int(q_values.argmax().item())]

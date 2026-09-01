"""Frozen inference-only Task-2 DQN baseline."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT")
MODEL_FILE = "dqn_task2_visit_count_v1_7000ep_seed11.pt"

WALL_CHANNEL = 0
CRATE_CHANNEL = 1
COIN_CHANNEL = 2
SELF_CHANNEL = 3
BOMB_TIMER_CHANNEL = 4
EXPLOSION_CHANNEL = 5
DANGER_CHANNEL = 6
BOMB_AVAILABLE_CHANNEL = 7
VISIT_COUNT_CHANNEL = 8

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
VISIT_COUNT_NORMALIZER = 10.0

ACTION_DELTAS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


class FrozenTask2DQN(nn.Module):
    """Architecture used by the frozen nine-channel Task-2 policy."""

    def __init__(self) -> None:
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.convolutional(states))


def setup(self) -> None:
    """Load the frozen policy for greedy CPU inference."""
    self.device = torch.device("cpu")
    self.online_network = FrozenTask2DQN().to(self.device)

    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"Missing frozen Task-2 model: {MODEL_FILE}")

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
    """Choose the highest-valued legal action."""
    features = _features_with_visit_count(self, game_state)
    action_mask = _legal_action_mask(features)
    state_tensor = torch.from_numpy(features).unsqueeze(0).float()

    with torch.inference_mode():
        q_values = self.online_network(state_tensor)[0]
        mask = torch.from_numpy(action_mask)
        q_values = q_values.masked_fill(~mask, -torch.inf)

    return ACTIONS[int(q_values.argmax().item())]


def _features_with_visit_count(self, game_state: dict) -> np.ndarray:
    """Add the frozen linear visit-count channel."""
    field = game_state["field"]
    round_number = game_state.get("round")
    if self.visit_round != round_number or self.visit_counts is None:
        self.visit_round = round_number
        self.visit_counts = np.zeros_like(field, dtype=np.float32)

    self_x, self_y = game_state["self"][3]
    self.visit_counts[self_x, self_y] += 1.0
    return _state_to_features(game_state, self.visit_counts)


def _state_to_features(
    game_state: dict,
    visit_counts: np.ndarray,
) -> np.ndarray:
    """Encode the board exactly as during best-model training."""
    field = game_state["field"]
    width, height = field.shape
    features = np.zeros((9, height, width), dtype=np.float32)
    features[WALL_CHANNEL] = (field == -1).T
    features[CRATE_CHANNEL] = (field == 1).T

    for x, y in game_state["coins"]:
        features[COIN_CHANNEL, y, x] = 1.0

    self_x, self_y = game_state["self"][3]
    features[SELF_CHANNEL, self_y, self_x] = 1.0

    for (bomb_x, bomb_y), timer in game_state.get("bombs", []):
        urgency = (BOMB_TIMER - min(max(int(timer), 0), BOMB_TIMER) + 1) / (
            BOMB_TIMER + 1
        )
        features[BOMB_TIMER_CHANNEL, bomb_y, bomb_x] = urgency
        for danger_x, danger_y in _blast_coordinates(
            field,
            (bomb_x, bomb_y),
        ):
            features[DANGER_CHANNEL, danger_y, danger_x] = max(
                features[DANGER_CHANNEL, danger_y, danger_x],
                urgency,
            )

    explosion_map = np.asarray(
        game_state.get("explosion_map", np.zeros_like(field)),
        dtype=np.float32,
    )
    features[EXPLOSION_CHANNEL] = np.clip(
        explosion_map / max(EXPLOSION_TIMER - 1, 1),
        0.0,
        1.0,
    ).T
    features[DANGER_CHANNEL] = np.maximum(
        features[DANGER_CHANNEL],
        features[EXPLOSION_CHANNEL],
    )
    features[BOMB_AVAILABLE_CHANNEL].fill(float(game_state["self"][2]))
    features[VISIT_COUNT_CHANNEL] = np.clip(
        visit_counts / VISIT_COUNT_NORMALIZER,
        0.0,
        1.0,
    ).T
    return features


def _blast_coordinates(
    field: np.ndarray,
    bomb_position: tuple[int, int],
) -> list[tuple[int, int]]:
    """Match the framework blast geometry: only stone walls stop it."""
    bomb_x, bomb_y = bomb_position
    width, height = field.shape
    coordinates = [(bomb_x, bomb_y)]
    for delta_x, delta_y in ACTION_DELTAS:
        for distance in range(1, BOMB_POWER + 1):
            x = bomb_x + delta_x * distance
            y = bomb_y + delta_y * distance
            if not (0 <= x < width and 0 <= y < height):
                break
            if field[x, y] == -1:
                break
            coordinates.append((x, y))
    return coordinates


def _legal_action_mask(features: np.ndarray) -> np.ndarray:
    """Mask movements blocked by walls, crates or bombs."""
    mask = np.ones(len(ACTIONS), dtype=np.bool_)
    self_y, self_x = np.argwhere(features[SELF_CHANNEL] > 0.5)[0]
    height, width = features.shape[1:]
    blocked = (
        (features[WALL_CHANNEL] > 0.5)
        | (features[CRATE_CHANNEL] > 0.5)
        | (features[BOMB_TIMER_CHANNEL] > 0.0)
    )
    for action_index, (delta_x, delta_y) in enumerate(ACTION_DELTAS):
        target_x = int(self_x + delta_x)
        target_y = int(self_y + delta_y)
        mask[action_index] = (
            0 <= target_x < width
            and 0 <= target_y < height
            and not blocked[target_y, target_x]
        )
    mask[4] = bool(features[BOMB_AVAILABLE_CHANNEL].max() > 0.5)
    mask[5] = True
    return mask

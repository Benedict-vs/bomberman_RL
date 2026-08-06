"""Create CNN input channels from a coin-heaven game state."""

from __future__ import annotations

import numpy as np


N_CHANNELS = 3

WALL_CHANNEL = 0
COIN_CHANNEL = 1
SELF_CHANNEL = 2


def state_to_features(game_state: dict | None) -> np.ndarray | None:
    """Return a float32 array with shape (3, height, width)."""
    if game_state is None:
        return None

    field = game_state["field"]
    width, height = field.shape

    features = np.zeros(
        (N_CHANNELS, height, width),
        dtype=np.float32,
    )

    # Framework coordinates: field[x, y].
    # CNN coordinates: features[channel, y, x].
    features[WALL_CHANNEL] = (field == -1).T

    for x, y in game_state["coins"]:
        features[COIN_CHANNEL, y, x] = 1.0

    self_x, self_y = game_state["self"][3]
    features[SELF_CHANNEL, self_y, self_x] = 1.0

    return features
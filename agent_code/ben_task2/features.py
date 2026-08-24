"""Create CNN input channels from a task-2 game state."""

from __future__ import annotations

import numpy as np


N_CHANNELS = 8

WALL_CHANNEL = 0
CRATE_CHANNEL = 1
COIN_CHANNEL = 2
SELF_CHANNEL = 3
BOMB_TIMER_CHANNEL = 4
EXPLOSION_CHANNEL = 5
DANGER_CHANNEL = 6
BOMB_AVAILABLE_CHANNEL = 7

BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2

ACTION_DELTAS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def state_to_features(game_state: dict | None) -> np.ndarray | None:
    """Return a float32 array with shape (8, height, width)."""
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
    features[CRATE_CHANNEL] = (field == 1).T

    for x, y in game_state["coins"]:
        features[COIN_CHANNEL, y, x] = 1.0

    self_x, self_y = game_state["self"][3]
    features[SELF_CHANNEL, self_y, self_x] = 1.0

    for (bomb_x, bomb_y), timer in game_state.get("bombs", []):
        urgency = _timer_urgency(timer)
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
    normalized_explosions = np.clip(
        explosion_map / max(EXPLOSION_TIMER - 1, 1),
        0.0,
        1.0,
    )
    features[EXPLOSION_CHANNEL] = normalized_explosions.T
    features[DANGER_CHANNEL] = np.maximum(
        features[DANGER_CHANNEL],
        features[EXPLOSION_CHANNEL],
    )

    bomb_available = bool(game_state["self"][2])
    features[BOMB_AVAILABLE_CHANNEL].fill(float(bomb_available))

    return features


def legal_action_mask(features: np.ndarray) -> np.ndarray:
    """Return legal actions in UP, RIGHT, DOWN, LEFT, BOMB, WAIT order."""
    mask = np.ones(6, dtype=np.bool_)
    self_positions = np.argwhere(features[SELF_CHANNEL] > 0.5)

    # Terminal replay states are all zero and never bootstrap. Returning
    # a valid mask still keeps the masked maximum numerically finite.
    if len(self_positions) == 0:
        return mask

    if len(self_positions) != 1:
        raise ValueError("Expected exactly one agent position in features.")

    self_y, self_x = self_positions[0]
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


def _timer_urgency(timer: int) -> float:
    """Map bomb timers to (0, 1], with larger values more urgent."""
    clipped_timer = min(max(int(timer), 0), BOMB_TIMER)
    return (BOMB_TIMER - clipped_timer + 1) / (BOMB_TIMER + 1)


def _blast_coordinates(
    field: np.ndarray,
    bomb_position: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return tiles hit by a bomb, matching the framework geometry."""
    bomb_x, bomb_y = bomb_position
    width, height = field.shape
    coordinates = [(bomb_x, bomb_y)]

    directions = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    for delta_x, delta_y in directions:
        for distance in range(1, BOMB_POWER + 1):
            x = bomb_x + delta_x * distance
            y = bomb_y + delta_y * distance

            if not (0 <= x < width and 0 <= y < height):
                break

            if field[x, y] == -1:
                break

            coordinates.append((x, y))

    return coordinates

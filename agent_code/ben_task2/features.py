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

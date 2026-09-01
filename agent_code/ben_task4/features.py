"""Create CNN input channels for Ben's Task-4 DQN agent."""

from __future__ import annotations

import numpy as np


N_CHANNELS = 11
N_ALIGNMENT_CHANNELS = 12
N_ESCAPE_CHANNELS = 11

WALL_CHANNEL = 0
CRATE_CHANNEL = 1
COIN_CHANNEL = 2
SELF_CHANNEL = 3
BOMB_TIMER_CHANNEL = 4
EXPLOSION_CHANNEL = 5
DANGER_CHANNEL = 6
BOMB_AVAILABLE_CHANNEL = 7
VISIT_COUNT_CHANNEL = 8
ESCAPE_TILES_CHANNEL = 9
OPPONENT_CHANNEL = 10
OPPONENT_BOMB_ALIGNMENT_CHANNEL = 11
VISIT_COUNT_NORMALIZER = 10.0
VISIT_COUNT_MAX = 400.0
VISIT_COUNT_ENCODING = "log_400"

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
    """Return Task-4 board channels with optional bomb alignment."""
    if game_state is None:
        return None

    field = game_state["field"]
    width, height = field.shape

    escape_feature_mode = game_state.get("escape_feature_mode")
    alignment_mode = game_state.get("opponent_alignment_mode", "disabled")
    if alignment_mode not in {"disabled", "zero", "enabled"}:
        raise ValueError(f"Unknown opponent-alignment mode: {alignment_mode}")
    channel_count = (
        N_ALIGNMENT_CHANNELS
        if alignment_mode in {"zero", "enabled"}
        else N_CHANNELS
    )
    features = np.zeros(
        (channel_count, height, width),
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

    for other in game_state.get("others", []):
        other_x, other_y = other[3]
        features[OPPONENT_CHANNEL, other_y, other_x] = 1.0

    if alignment_mode == "enabled":
        opponent_positions = {
            tuple(other[3]) for other in game_state.get("others", [])
        }
        for source_x in range(width):
            for source_y in range(height):
                if field[source_x, source_y] != 0:
                    continue
                if opponent_positions.intersection(
                    _blast_coordinates(field, (source_x, source_y))
                ):
                    features[
                        OPPONENT_BOMB_ALIGNMENT_CHANNEL,
                        source_y,
                        source_x,
                    ] = 1.0

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

    visit_counts = np.asarray(
        game_state.get("visit_counts", np.zeros_like(field)),
        dtype=np.float32,
    )
    visit_count_encoding = game_state.get(
        "visit_count_encoding",
        VISIT_COUNT_ENCODING,
    )
    if visit_count_encoding == "linear_10":
        encoded_visit_counts = np.clip(
            visit_counts / VISIT_COUNT_NORMALIZER,
            0.0,
            1.0,
        )
    elif visit_count_encoding == "log_400":
        encoded_visit_counts = np.log1p(
            np.clip(visit_counts, 0.0, VISIT_COUNT_MAX)
        ) / np.log1p(VISIT_COUNT_MAX)
    else:
        raise ValueError(
            f"Unknown visit-count encoding: {visit_count_encoding}"
        )

    features[VISIT_COUNT_CHANNEL] = encoded_visit_counts.T

    if escape_feature_mode == "reachable_safe_tiles":
        features[ESCAPE_TILES_CHANNEL] = _reachable_escape_tiles(
            field,
            (self_x, self_y),
            game_state.get("bombs", []),
            features[DANGER_CHANNEL],
            bomb_available,
            {other[3] for other in game_state.get("others", [])},
        ).T
    elif escape_feature_mode not in (None, "zero"):
        raise ValueError(
            f"Unknown escape-feature mode: {escape_feature_mode}"
        )

    return features


def _reachable_escape_tiles(
    field: np.ndarray,
    start: tuple[int, int],
    bombs: list[tuple[tuple[int, int], int]],
    danger: np.ndarray,
    bomb_available: bool,
    opponent_positions: set[tuple[int, int]],
) -> np.ndarray:
    """Mark safe endpoints reachable before a hypothetical bomb explodes."""
    escape_tiles = np.zeros_like(field, dtype=np.float32)
    if not bomb_available:
        return escape_tiles

    hypothetical_blast = set(_blast_coordinates(field, start))
    bomb_positions = {position for position, _timer in bombs}
    blocked = (field != 0).copy()
    for opponent_x, opponent_y in opponent_positions:
        blocked[opponent_x, opponent_y] = True

    distances = {start: 0}
    frontier = [start]

    while frontier:
        x, y = frontier.pop(0)
        distance = distances[(x, y)]

        if (
            (x, y) not in hypothetical_blast
            and danger[y, x] <= 0.0
        ):
            escape_tiles[x, y] = 1.0

        if distance >= BOMB_TIMER:
            continue

        for delta_x, delta_y in ACTION_DELTAS:
            target = (x + delta_x, y + delta_y)
            target_x, target_y = target
            if (
                0 <= target_x < field.shape[0]
                and 0 <= target_y < field.shape[1]
                and target not in distances
                and target not in bomb_positions
                and not blocked[target_x, target_y]
            ):
                distances[target] = distance + 1
                frontier.append(target)

    return escape_tiles


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
        | (features[OPPONENT_CHANNEL] > 0.5)
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

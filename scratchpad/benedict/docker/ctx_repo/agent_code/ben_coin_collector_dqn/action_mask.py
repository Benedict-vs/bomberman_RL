"""Legal-action masking for the coin-heaven agent."""

from __future__ import annotations

import torch

from .features import SELF_CHANNEL, WALL_CHANNEL


MOVEMENT_DELTAS = (
    (0, -1),   # UP
    (1, 0),    # RIGHT
    (0, 1),    # DOWN
    (-1, 0),   # LEFT
)


def legal_action_mask(states: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask for UP, RIGHT, DOWN, LEFT and WAIT."""
    if states.ndim != 4:
        raise ValueError(
            "Expected states with shape "
            "(batch, channels, height, width)."
        )

    batch_size, _, height, width = states.shape

    flat_positions = (
        states[:, SELF_CHANNEL]
        .flatten(start_dim=1)
        .argmax(dim=1)
    )

    self_y = flat_positions // width
    self_x = flat_positions % width

    deltas = torch.tensor(
        MOVEMENT_DELTAS,
        dtype=torch.long,
        device=states.device,
    )

    target_x = self_x.unsqueeze(1) + deltas[:, 0]
    target_y = self_y.unsqueeze(1) + deltas[:, 1]

    inside_board = (
        (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
    )

    safe_x = target_x.clamp(0, width - 1)
    safe_y = target_y.clamp(0, height - 1)

    batch_indices = torch.arange(
        batch_size,
        device=states.device,
    ).unsqueeze(1)

    target_is_wall = (
        states[
            batch_indices,
            WALL_CHANNEL,
            safe_y,
            safe_x,
        ]
        > 0.5
    )

    legal_movements = inside_board & ~target_is_wall

    wait_is_legal = torch.ones(
        (batch_size, 1),
        dtype=torch.bool,
        device=states.device,
    )

    return torch.cat(
        (legal_movements, wait_is_legal),
        dim=1,
    )
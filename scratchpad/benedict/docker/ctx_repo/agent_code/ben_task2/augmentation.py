"""Symmetry augmentation for task-2 transitions."""

from __future__ import annotations

import random

import numpy as np

from .model import ACTIONS
from .replay_buffer import Transition


_ACTION_VECTORS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}

_VECTOR_ACTIONS = {
    vector: action
    for action, vector in _ACTION_VECTORS.items()
}


def transform_transition(
    transition: Transition,
    rotations: int,
    reflect: bool,
) -> Transition:
    """Apply one board symmetry consistently to a transition."""
    normalized_rotations = rotations % 4

    transformed_state = _transform_board(
        transition.state,
        rotations=normalized_rotations,
        reflect=reflect,
    )

    transformed_next_state = _transform_board(
        transition.next_state,
        rotations=normalized_rotations,
        reflect=reflect,
    )

    transformed_action = _transform_action(
        transition.action,
        rotations=normalized_rotations,
        reflect=reflect,
    )

    return Transition(
        state=transformed_state,
        action=transformed_action,
        reward=transition.reward,
        next_state=transformed_next_state,
        done=transition.done,
    )


def randomly_transform_transition(
    transition: Transition,
) -> Transition:
    """Apply one uniformly sampled rotation and reflection."""
    rotations = random.randrange(4)
    reflect = bool(
        random.getrandbits(1)
    )

    return transform_transition(
        transition,
        rotations=rotations,
        reflect=reflect,
    )


def _transform_board(
    board: np.ndarray,
    rotations: int,
    reflect: bool,
) -> np.ndarray:
    """Rotate and optionally reflect a channel-first board."""
    transformed = np.rot90(
        board,
        k=rotations,
        axes=(-2, -1),
    )

    if reflect:
        transformed = np.flip(
            transformed,
            axis=-1,
        )

    # Rotations and flips can produce arrays with negative strides.
    # A separate C-contiguous copy is safe for torch.from_numpy.
    return np.array(
        transformed,
        dtype=board.dtype,
        copy=True,
        order="C",
    )


def _transform_action(
    action_index: int,
    rotations: int,
    reflect: bool,
) -> int:
    """Transform one action consistently with the board."""
    action = ACTIONS[action_index]

    if action in ("BOMB", "WAIT"):
        return action_index

    delta_y, delta_x = _ACTION_VECTORS[action]

    for _ in range(rotations):
        # 90-degree counterclockwise rotation.
        delta_y, delta_x = (
            -delta_x,
            delta_y,
        )

    if reflect:
        # Horizontal reflection reverses the x direction.
        delta_x = -delta_x

    transformed_action = _VECTOR_ACTIONS[
        (delta_y, delta_x)
    ]

    return ACTIONS.index(
        transformed_action
    )

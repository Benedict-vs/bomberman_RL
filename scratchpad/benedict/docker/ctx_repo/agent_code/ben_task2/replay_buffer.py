from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One observed environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size buffer storing recent transitions."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")

        self._transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def capacity(self) -> int:
        return self._transitions.maxlen

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
    ) -> None:
        """Store one transition."""
        if next_state is None:
            next_state = np.zeros_like(state)

        transition = Transition(
            state=state.copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.copy(),
            done=bool(done),
        )

        self._transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a uniformly sampled batch without replacement."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from a buffer containing {len(self)}."
            )

        return random.sample(
            list(self._transitions),
            batch_size,
        )

    def mark_last_terminal(self) -> None:
        """Convert the most recent transition into a terminal transition."""
        if not self._transitions:
            raise ValueError(
                "Cannot mark an empty replay buffer as terminal."
            )

        last = self._transitions[-1]

        self._transitions[-1] = Transition(
            state=last.state,
            action=last.action,
            reward=last.reward,
            next_state=np.zeros_like(last.next_state),
            done=True,
        )
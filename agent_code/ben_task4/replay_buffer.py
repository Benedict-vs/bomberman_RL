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
    n_steps: int = 1
    event_tags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BombOutcomeExample:
    """One resolved own-bomb state and its observed outcome class."""

    state: np.ndarray
    outcome: int  # 0=neutral, 1=kill, 2=self-death


class BombOutcomeBuffer:
    """Small uniform store for resolved own-bomb auxiliary examples."""

    def __init__(self, capacity: int = 20_000) -> None:
        self._examples: deque[BombOutcomeExample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._examples)

    def append(self, state: np.ndarray, outcome: int) -> None:
        if outcome not in {0, 1, 2}:
            raise ValueError("Bomb outcome must be neutral, kill, or self-death.")
        self._examples.append(BombOutcomeExample(state.copy(), int(outcome)))

    def sample(self, batch_size: int) -> list[BombOutcomeExample]:
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("Invalid bomb-outcome batch size.")
        return random.sample(list(self._examples), batch_size)


class ReplayBuffer:
    """Fixed-size buffer storing recent transitions."""

    def __init__(
        self,
        capacity: int,
        event_balance: bool = False,
        prioritized: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")

        self._transitions: deque[Transition] = deque(maxlen=capacity)
        self.event_balance = bool(event_balance)
        self.prioritized = bool(prioritized)
        self._priorities: deque[float] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def capacity(self) -> int:
        return self._transitions.maxlen

    def state_dict(self) -> dict[str, object]:
        """Return the complete replay state needed for an exact local resume."""
        return {
            "capacity": self.capacity,
            "event_balance": self.event_balance,
            "prioritized": self.prioritized,
            "transitions": list(self._transitions),
            "priorities": list(self._priorities),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReplayBuffer":
        """Restore a buffer saved by :meth:`state_dict`."""
        capacity = int(state["capacity"])
        buffer = cls(
            capacity=capacity,
            event_balance=bool(state["event_balance"]),
            prioritized=bool(state["prioritized"]),
        )
        transitions = list(state["transitions"])
        priorities = list(state["priorities"])
        if len(transitions) != len(priorities) or len(transitions) > capacity:
            raise ValueError("Invalid replay-buffer training state.")
        if not all(isinstance(transition, Transition) for transition in transitions):
            raise ValueError("Invalid transition in replay-buffer training state.")
        buffer._transitions.extend(transitions)
        buffer._priorities.extend(float(priority) for priority in priorities)
        return buffer

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
        n_steps: int = 1,
        event_tags: frozenset[str] = frozenset(),
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
            n_steps=int(n_steps),
            event_tags=frozenset(event_tags),
        )

        self._transitions.append(transition)
        self._priorities.append(max(self._priorities, default=1.0))

    def sample(self, batch_size: int) -> list[Transition]:
        """Return a uniformly sampled batch without replacement."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from a buffer containing {len(self)}."
            )

        transitions = list(self._transitions)
        if not self.event_balance:
            return random.sample(transitions, batch_size)

        selected_indices: list[int] = []
        for tag, fraction in (("kill", 0.20), ("death", 0.15)):
            candidates = [
                index
                for index, transition in enumerate(transitions)
                if tag in transition.event_tags
                and index not in selected_indices
            ]
            target = min(int(batch_size * fraction), len(candidates))
            selected_indices.extend(random.sample(candidates, target))

        selected = set(selected_indices)
        remaining = [
            index for index in range(len(transitions)) if index not in selected
        ]
        selected_indices.extend(
            random.sample(remaining, batch_size - len(selected_indices))
        )
        return [transitions[index] for index in selected_indices]

    def sample_prioritized(
        self,
        batch_size: int,
        beta: float,
        alpha: float = 0.6,
    ) -> tuple[list[Transition], list[int], np.ndarray]:
        """Sample with TD-error priorities and normalized IS weights."""
        if not self.prioritized:
            raise ValueError("Prioritized sampling is disabled for this buffer.")
        if batch_size <= 0 or not self._transitions:
            raise ValueError("Batch size must be positive and buffer non-empty.")
        if not 0.0 <= beta <= 1.0 or alpha < 0.0:
            raise ValueError("Invalid PER alpha or beta.")

        priorities = np.asarray(self._priorities, dtype=np.float64)
        probabilities = priorities**alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(
            len(self._transitions), size=batch_size, replace=True, p=probabilities
        )
        weights = (len(self._transitions) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        transitions = list(self._transitions)
        return (
            [transitions[int(index)] for index in indices],
            [int(index) for index in indices],
            weights.astype(np.float32),
        )

    def update_priorities(
        self, indices: list[int], td_errors: np.ndarray, epsilon: float = 1e-5
    ) -> None:
        """Update sampled priorities after an optimization step."""
        if not self.prioritized:
            raise ValueError("Prioritized replay is disabled for this buffer.")
        if len(indices) != len(td_errors) or epsilon <= 0.0:
            raise ValueError("Invalid priority update.")
        priorities = list(self._priorities)
        for index, error in zip(indices, td_errors, strict=True):
            if not 0 <= index < len(priorities):
                raise IndexError("Replay priority index out of range.")
            priorities[index] = max(float(abs(error)) + epsilon, epsilon)
        self._priorities = deque(priorities, maxlen=self.capacity)

    def mark_last_terminal(self, reward_adjustment: float = 0.0) -> None:
        """Convert the most recent transition into a terminal transition.

        ``reward_adjustment`` lets the caller remove a potential's
        non-terminal endpoint term once the framework reports that the round
        actually ended after the transition was first stored.
        """
        if not self._transitions:
            raise ValueError(
                "Cannot mark an empty replay buffer as terminal."
            )

        last = self._transitions[-1]

        self._transitions[-1] = Transition(
            state=last.state,
            action=last.action,
            reward=last.reward + float(reward_adjustment),
            next_state=np.zeros_like(last.next_state),
            done=True,
            n_steps=last.n_steps,
            event_tags=last.event_tags,
        )

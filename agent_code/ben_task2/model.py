"""Neural-network model for the task-2 DQN agent."""

from __future__ import annotations

import torch
from torch import nn


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT")

N_INPUT_CHANNELS = 8
N_ACTIONS = len(ACTIONS)


class CoinCollectorDQN(nn.Module):
    """Small CNN mapping a board state to one Q-value per action."""

    def __init__(self) -> None:
        super().__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(
                in_channels=N_INPUT_CHANNELS,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape (batch_size, number_of_actions)."""
        encoded_states = self.convolutional(states)
        return self.q_head(encoded_states)

"""Neural-network model for Ben's Task-4 DQN agent."""

from __future__ import annotations

import torch
from torch import nn


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT")

N_INPUT_CHANNELS = 11
N_ACTIONS = len(ACTIONS)


class CoinCollectorDQN(nn.Module):
    """Small CNN mapping a board state to one Q-value per action."""

    def __init__(self, input_channels: int = N_INPUT_CHANNELS) -> None:
        super().__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
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


class DuelingCoinCollectorDQN(nn.Module):
    """Dueling variant with a value stream and an action-advantage stream."""

    def __init__(self, input_channels: int = N_INPUT_CHANNELS) -> None:
        super().__init__()
        self.convolutional = CoinCollectorDQN(input_channels).convolutional
        self.feature_head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.ReLU()
        )
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, N_ACTIONS)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.feature_head(self.convolutional(states))
        advantages = self.advantage_head(features)
        return self.value_head(features) + advantages - advantages.mean(
            dim=1, keepdim=True
        )


class OpponentPredictionDQN(nn.Module):
    """Classic Q network with an auxiliary future-opponent occupancy head."""

    def __init__(self, input_channels: int = N_INPUT_CHANNELS) -> None:
        super().__init__()
        base = CoinCollectorDQN(input_channels)
        self.convolutional = base.convolutional
        self.q_head = base.q_head
        self.opponent_prediction_head = nn.Linear(128, 17 * 17)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.convolutional(states))

    def forward_with_opponent_prediction(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.convolutional(states)
        hidden = self.q_head[2](self.q_head[1](self.q_head[0](encoded)))
        return self.q_head[3](hidden), self.opponent_prediction_head(hidden).view(
            -1, 17, 17
        )


class BombOutcomeDQN(nn.Module):
    """Classic Q network plus a three-class own-bomb outcome head."""

    def __init__(self, input_channels: int = N_INPUT_CHANNELS) -> None:
        super().__init__()
        base = CoinCollectorDQN(input_channels)
        self.convolutional = base.convolutional
        self.q_head = base.q_head
        self.bomb_outcome_head = nn.Linear(128, 3)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.convolutional(states))

    def forward_with_bomb_outcome(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.convolutional(states)
        hidden = self.q_head[2](self.q_head[1](self.q_head[0](encoded)))
        return self.q_head[3](hidden), self.bomb_outcome_head(hidden)


def opponent_prediction_state_dict_from_classic(
    classic_state_dict: dict[str, torch.Tensor], input_channels: int = N_INPUT_CHANNELS
) -> dict[str, torch.Tensor]:
    """Create an auxiliary model whose Q head exactly equals the classic model."""
    classic = CoinCollectorDQN(input_channels)
    classic.load_state_dict(classic_state_dict)
    auxiliary = OpponentPredictionDQN(input_channels)
    auxiliary.convolutional.load_state_dict(classic.convolutional.state_dict())
    auxiliary.q_head.load_state_dict(classic.q_head.state_dict())
    return auxiliary.state_dict()


def bomb_outcome_state_dict_from_classic(
    classic_state_dict: dict[str, torch.Tensor], input_channels: int = N_INPUT_CHANNELS
) -> dict[str, torch.Tensor]:
    """Create a bomb-outcome model whose Q-values equal the classic model."""
    classic = CoinCollectorDQN(input_channels)
    classic.load_state_dict(classic_state_dict)
    auxiliary = BombOutcomeDQN(input_channels)
    auxiliary.convolutional.load_state_dict(classic.convolutional.state_dict())
    auxiliary.q_head.load_state_dict(classic.q_head.state_dict())
    return auxiliary.state_dict()


def dueling_state_dict_from_classic(
    classic_state_dict: dict[str, torch.Tensor], input_channels: int = N_INPUT_CHANNELS
) -> dict[str, torch.Tensor]:
    """Convert a classic Q head into an exactly Q-preserving dueling head."""
    classic = CoinCollectorDQN(input_channels)
    classic.load_state_dict(classic_state_dict)
    dueling = DuelingCoinCollectorDQN(input_channels)
    dueling.convolutional.load_state_dict(classic.convolutional.state_dict())
    dueling.feature_head.load_state_dict(
        nn.Sequential(*list(classic.q_head.children())[:3]).state_dict()
    )
    final = classic.q_head[-1]
    dueling.advantage_head.load_state_dict(final.state_dict())
    with torch.no_grad():
        dueling.value_head.weight.copy_(final.weight.mean(dim=0, keepdim=True))
        dueling.value_head.bias.copy_(final.bias.mean(dim=0, keepdim=True))
    return dueling.state_dict()

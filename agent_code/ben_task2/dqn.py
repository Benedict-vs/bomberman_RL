"""Core Deep Q-Learning update functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .features import legal_action_mask
from .replay_buffer import Transition


def optimize_dqn(
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    gamma: float,
    device: torch.device,
) -> float:
    """Perform one DQN optimization step and return the loss."""
    if not transitions:
        raise ValueError(
            "Cannot optimize an empty transition batch."
        )

    states = torch.from_numpy(
        np.stack(
            [
                transition.state
                for transition in transitions
            ]
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )

    actions = torch.tensor(
        [
            transition.action
            for transition in transitions
        ],
        dtype=torch.long,
        device=device,
    )

    rewards = torch.tensor(
        [
            transition.reward
            for transition in transitions
        ],
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.from_numpy(
        np.stack(
            [
                transition.next_state
                for transition in transitions
            ]
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )

    dones = torch.tensor(
        [
            transition.done
            for transition in transitions
        ],
        dtype=torch.float32,
        device=device,
    )

    predicted_q_values = online_network(
        states
    )

    selected_q_values = predicted_q_values.gather(
        dim=1,
        index=actions.unsqueeze(1),
    ).squeeze(1)

    with torch.no_grad():
        next_action_masks = torch.from_numpy(
            np.stack(
                [
                    legal_action_mask(transition.next_state)
                    for transition in transitions
                ]
            )
        ).to(device=device)

        masked_next_q_values = target_network(
            next_states
        ).masked_fill(
            ~next_action_masks,
            -torch.inf,
        )

        next_q_values = masked_next_q_values.max(dim=1).values

        target_q_values = (
            rewards
            + gamma
            * (1.0 - dones)
            * next_q_values
        )

    loss = nn.functional.smooth_l1_loss(
        selected_q_values,
        target_q_values,
    )

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    return float(loss.item())


def update_target_network(
    online_network: nn.Module,
    target_network: nn.Module,
) -> None:
    """Copy all online-network parameters to the target network."""
    target_network.load_state_dict(
        online_network.state_dict()
    )


def soft_update_target_network(
    online_network: nn.Module,
    target_network: nn.Module,
    tau: float,
) -> None:
    """Move target parameters a fraction tau toward online parameters."""
    if not 0.0 < tau <= 1.0:
        raise ValueError(
            "Soft-update tau must satisfy 0 < tau <= 1."
        )

    with torch.no_grad():
        parameter_pairs = zip(
            online_network.parameters(),
            target_network.parameters(),
            strict=True,
        )

        for online_parameter, target_parameter in parameter_pairs:
            target_parameter.mul_(
                1.0 - tau
            )

            target_parameter.add_(
                online_parameter,
                alpha=tau,
            )

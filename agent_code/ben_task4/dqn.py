"""Core Deep Q-Learning update functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .features import legal_action_mask
from .features import OPPONENT_CHANNEL
from .replay_buffer import BombOutcomeExample, Transition


def optimize_dqn(
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    gamma: float,
    device: torch.device,
    double_dqn: bool = False,
    importance_weights: np.ndarray | None = None,
    return_td_errors: bool = False,
    auxiliary_opponent_prediction: bool = False,
    auxiliary_scale: float = 0.1,
    bomb_outcome_examples: Sequence[BombOutcomeExample] | None = None,
    auxiliary_bomb_outcome: bool = False,
) -> float | tuple[float, np.ndarray]:
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

    n_steps = torch.tensor(
        [transition.n_steps for transition in transitions],
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

        target_next_q_values = target_network(next_states).masked_fill(
            ~next_action_masks, -torch.inf
        )
        if double_dqn:
            online_next_actions = online_network(next_states).masked_fill(
                ~next_action_masks, -torch.inf
            ).argmax(dim=1)
            next_q_values = target_next_q_values.gather(
                dim=1, index=online_next_actions.unsqueeze(1)
            ).squeeze(1)
        else:
            next_q_values = target_next_q_values.max(dim=1).values

        target_q_values = (
            rewards
            + torch.pow(gamma, n_steps)
            * (1.0 - dones)
            * next_q_values
        )

    per_item_loss = nn.functional.smooth_l1_loss(
        selected_q_values,
        target_q_values,
        reduction="none",
    )
    if importance_weights is None:
        loss = per_item_loss.mean()
    else:
        weights = torch.as_tensor(
            importance_weights, dtype=torch.float32, device=device
        )
        if weights.shape != per_item_loss.shape:
            raise ValueError("Importance weights must match batch size.")
        loss = (weights * per_item_loss).mean()

    if auxiliary_opponent_prediction:
        if not hasattr(online_network, "forward_with_opponent_prediction"):
            raise ValueError("Auxiliary head missing from online network.")
        _q_values, prediction_logits = online_network.forward_with_opponent_prediction(
            states
        )
        opponent_target = next_states[:, OPPONENT_CHANNEL]
        positive_weight = torch.tensor(32.0, device=device)
        auxiliary_loss = nn.functional.binary_cross_entropy_with_logits(
            prediction_logits, opponent_target, pos_weight=positive_weight
        )
        loss = loss + auxiliary_scale * auxiliary_loss

    if auxiliary_bomb_outcome and bomb_outcome_examples:
        if not hasattr(online_network, "forward_with_bomb_outcome"):
            raise ValueError("Bomb-outcome head missing from online network.")
        outcome_states = torch.from_numpy(np.stack([
            example.state for example in bomb_outcome_examples
        ])).to(device=device, dtype=torch.float32)
        outcome_targets = torch.tensor(
            [example.outcome for example in bomb_outcome_examples],
            dtype=torch.long, device=device,
        )
        _q_values, outcome_logits = online_network.forward_with_bomb_outcome(
            outcome_states
        )
        # Kills are rare; weighting prevents the neutral class from becoming
        # the trivial auxiliary solution.
        class_weights = torch.tensor([1.0, 8.0, 3.0], device=device)
        auxiliary_loss = nn.functional.cross_entropy(
            outcome_logits, outcome_targets, weight=class_weights
        )
        loss = loss + auxiliary_scale * auxiliary_loss

    td_errors = (target_q_values - selected_q_values).abs().detach().cpu().numpy()

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    result = float(loss.item())
    return (result, td_errors) if return_td_errors else result


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

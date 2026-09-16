"""Executable checks for the 2026-09-07 n-step design audit."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_code.ben_task4 import train
from agent_code.ben_task4.dqn import optimize_dqn
from agent_code.ben_task4.replay_buffer import ReplayBuffer, Transition


def state(value: float) -> np.ndarray:
    result = np.zeros((11, 17, 17), dtype=np.float32)
    result[3, 1, 1] = 1.0
    result[0] += value
    return result


def queue_trace(terminal_via_append: bool) -> list[Transition]:
    old_n = train.N_STEP_RETURN
    train.N_STEP_RETURN = 3
    agent = SimpleNamespace(n_step_queue=deque(), replay_buffer=ReplayBuffer(20))
    try:
        for index, reward in enumerate((1.0, 2.0, 3.0, 4.0)):
            done = terminal_via_append and index == 3
            train._append_n_step_transition(
                agent,
                state(index),
                index % 6,
                reward,
                None if done else state(index + 1),
                done,
            )
        if not terminal_via_append:
            train._finish_n_step_episode(agent)
        assert not agent.n_step_queue
        return list(agent.replay_buffer._transitions)
    finally:
        train.N_STEP_RETURN = old_n


class ConstantNetwork(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value.expand(states.shape[0], 6)


def check_bootstrap_exponent(n_steps: int) -> None:
    online = ConstantNetwork(0.0)
    target = ConstantNetwork(2.0)
    optimizer = torch.optim.SGD(online.parameters(), lr=0.0)
    transition = Transition(
        state=state(0),
        action=0,
        reward=0.0,
        next_state=state(1),
        done=False,
        n_steps=n_steps,
    )
    loss = optimize_dqn(
        online, target, optimizer, [transition], 0.99, torch.device("cpu")
    )
    target_value = (0.99**n_steps) * 2.0
    expected_huber = target_value - 0.5
    assert abs(loss - expected_huber) < 1e-6, (loss, expected_huber)


def main() -> None:
    expected_rewards = [
        1.0 + 0.99 * 2.0 + 0.99**2 * 3.0,
        2.0 + 0.99 * 3.0 + 0.99**2 * 4.0,
        3.0 + 0.99 * 4.0,
        4.0,
    ]
    for death in (False, True):
        transitions = queue_trace(death)
        assert len(transitions) == 4
        assert [item.n_steps for item in transitions] == [3, 3, 2, 1]
        assert [item.done for item in transitions] == [False, True, True, True]
        np.testing.assert_allclose(
            [item.reward for item in transitions], expected_rewards, rtol=1e-7
        )
        assert [item.action for item in transitions] == [0, 1, 2, 3]

    check_bootstrap_exponent(1)
    check_bootstrap_exponent(3)
    print("queue traces, shortened terminal returns, and gamma**n verified")


if __name__ == "__main__":
    main()

import unittest

import numpy as np
import torch
from torch import nn

from agent_code.ben_coin_collector_dqn.dqn import (
    optimize_dqn,
    soft_update_target_network,
    update_target_network,
)
from agent_code.ben_coin_collector_dqn.replay_buffer import Transition


class TinyQNetwork(nn.Module):
    """Controllable Q-network used only for unit tests."""

    def __init__(self, q_values: list[float]) -> None:
        super().__init__()

        self.q_values = nn.Parameter(
            torch.tensor(
                q_values,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        return self.q_values.unsqueeze(0).expand(
            states.shape[0],
            -1,
        )


class DQNUpdateTest(unittest.TestCase):
    @staticmethod
    def make_transition(
        action: int = 0,
        reward: float = 1.0,
        done: bool = False,
    ) -> Transition:
        state = np.zeros(
            (3, 17, 17),
            dtype=np.float32,
        )

        return Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=state.copy(),
            done=done,
        )

    def test_optimization_changes_online_network_only(self):
        online_network = TinyQNetwork([0.0] * 5)
        target_network = TinyQNetwork([0.0] * 5)

        optimizer = torch.optim.SGD(
            online_network.parameters(),
            lr=0.1,
        )

        target_before = (
            target_network.q_values.detach().clone()
        )

        loss = optimize_dqn(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            transitions=[
                self.make_transition()
            ],
            gamma=0.9,
            device=torch.device("cpu"),
        )

        self.assertGreater(
            loss,
            0.0,
        )

        self.assertFalse(
            torch.equal(
                online_network.q_values.detach(),
                torch.zeros(5),
            )
        )

        self.assertTrue(
            torch.equal(
                target_network.q_values.detach(),
                target_before,
            )
        )

        self.assertIsNone(
            target_network.q_values.grad
        )

    def test_terminal_transition_does_not_bootstrap(self):
        online_network = TinyQNetwork([0.0] * 5)
        target_network = TinyQNetwork([100.0] * 5)

        optimizer = torch.optim.SGD(
            online_network.parameters(),
            lr=0.0,
        )

        loss = optimize_dqn(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            transitions=[
                self.make_transition(
                    action=0,
                    reward=1.0,
                    done=True,
                )
            ],
            gamma=0.9,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(
            loss,
            0.5,
            places=6,
        )

    def test_target_update_copies_parameters(self):
        online_network = TinyQNetwork(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        target_network = TinyQNetwork([0.0] * 5)

        update_target_network(
            online_network,
            target_network,
        )

        self.assertTrue(
            torch.equal(
                online_network.q_values.detach(),
                target_network.q_values.detach(),
            )
        )

    def test_soft_target_update_blends_parameters(self):
        online_network = TinyQNetwork(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        target_network = TinyQNetwork(
            [0.0, 0.0, 0.0, 0.0, 0.0]
        )

        online_before = (
            online_network.q_values.detach().clone()
        )

        soft_update_target_network(
            online_network=online_network,
            target_network=target_network,
            tau=0.1,
        )

        expected_target = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            dtype=torch.float32,
        )

        self.assertTrue(
            torch.allclose(
                target_network.q_values.detach(),
                expected_target,
            )
        )

        self.assertTrue(
            torch.equal(
                online_network.q_values.detach(),
                online_before,
            )
        )

    def test_soft_target_update_rejects_invalid_tau(self):
        online_network = TinyQNetwork([1.0] * 5)
        target_network = TinyQNetwork([0.0] * 5)

        for invalid_tau in (
            0.0,
            -0.1,
            1.1,
        ):
            with self.subTest(
                tau=invalid_tau
            ):
                with self.assertRaises(ValueError):
                    soft_update_target_network(
                        online_network=online_network,
                        target_network=target_network,
                        tau=invalid_tau,
                    )

    def test_empty_batch_is_rejected(self):
        online_network = TinyQNetwork([0.0] * 5)
        target_network = TinyQNetwork([0.0] * 5)

        optimizer = torch.optim.SGD(
            online_network.parameters(),
            lr=0.1,
        )

        with self.assertRaises(ValueError):
            optimize_dqn(
                online_network=online_network,
                target_network=target_network,
                optimizer=optimizer,
                transitions=[],
                gamma=0.9,
                device=torch.device("cpu"),
            )

    def test_standard_dqn_uses_largest_target_value(self):
        online_network = TinyQNetwork(
            [0.0, 5.0, 0.0, 0.0, 0.0]
        )
        target_network = TinyQNetwork(
            [100.0, 2.0, 0.0, 0.0, 0.0]
        )

        optimizer = torch.optim.SGD(
            online_network.parameters(),
            lr=0.0,
        )

        loss = optimize_dqn(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            transitions=[
                self.make_transition(
                    action=0,
                    reward=1.0,
                    done=False,
                )
            ],
            gamma=0.5,
            device=torch.device("cpu"),
        )

        # Standard DQN uses the largest target-network value.
        # Bellman target: 1 + 0.5 * 100 = 51.
        # Huber loss between prediction 0 and target 51 = 50.5.
        self.assertAlmostEqual(
            loss,
            50.5,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
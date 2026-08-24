import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from agent_code.ben_task2 import callbacks
from agent_code.ben_task2.model import ACTIONS


class CallbacksTest(unittest.TestCase):
    @staticmethod
    def make_game_state(
        bomb_available: bool = False,
    ) -> dict:
        field = np.zeros((17, 17), dtype=np.int8)

        return {
            "field": field,
            "coins": [(5, 5)],
            "self": ("dqn-agent", 0, bomb_available, (1, 1)),
            "bombs": [],
            "explosion_map": np.zeros_like(field),
        }

    @staticmethod
    def setup_agent(train: bool = False) -> SimpleNamespace:
        agent = SimpleNamespace(
            train=train,
            logger=Mock(),
        )

        previous_directory = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                os.chdir(temporary_directory)
                callbacks.setup(agent)
        finally:
            os.chdir(previous_directory)

        return agent

    def test_setup_uses_cpu(self):
        agent = self.setup_agent(train=False)

        self.assertEqual(
            agent.device,
            torch.device("cpu"),
        )

        parameter_device = next(
            agent.online_network.parameters()
        ).device

        self.assertEqual(
            parameter_device.type,
            "cpu",
        )
        self.assertFalse(
            agent.online_network.training
        )

    def test_greedy_action_uses_largest_q_value(self):
        agent = self.setup_agent(train=False)

        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()

            # BOMB has the largest raw Q-value but is unavailable.
            # LEFT is the largest legal action.
            agent.online_network.q_head[-1].bias[4] = 3.0
            agent.online_network.q_head[-1].bias[3] = 2.0

        action = callbacks.act(
            agent,
            self.make_game_state(),
        )

        self.assertEqual(action, "LEFT")

    def test_exploration_uses_only_legal_actions(self):
        agent = self.setup_agent(train=True)
        agent.epsilon = 1.0

        selected_actions = {
            callbacks.act(
                agent,
                game_state=self.make_game_state(),
            )
            for _ in range(100)
        }

        self.assertTrue(selected_actions.issubset(set(ACTIONS)))
        self.assertNotIn("BOMB", selected_actions)

    def test_exploration_can_select_available_bomb(self):
        agent = self.setup_agent(train=True)
        agent.epsilon = 1.0

        selected_actions = {
            callbacks.act(
                agent,
                game_state=self.make_game_state(bomb_available=True),
            )
            for _ in range(200)
        }

        self.assertIn("BOMB", selected_actions)


if __name__ == "__main__":
    unittest.main()

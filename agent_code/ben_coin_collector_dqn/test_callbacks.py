import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from agent_code.ben_coin_collector_dqn import callbacks
from agent_code.ben_coin_collector_dqn.model import ACTIONS


class CallbacksTest(unittest.TestCase):
    @staticmethod
    def make_game_state() -> dict:
        field = np.zeros((17, 17), dtype=np.int8)

        return {
            "field": field,
            "coins": [(5, 5)],
            "self": ("dqn-agent", 0, False, (1, 1)),
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

        self.assertEqual(agent.device, torch.device("cpu"))

        parameter_device = next(
            agent.online_network.parameters()
        ).device

        self.assertEqual(parameter_device.type, "cpu")
        self.assertFalse(agent.online_network.training)

    def test_greedy_action_uses_largest_legal_q_value(self):
        agent = self.setup_agent(train=False)

        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()

            # LEFT, action index 3, is legal and has the largest Q-value.
            agent.online_network.q_head[-1].bias[3] = 2.0

        action = callbacks.act(
            agent,
            self.make_game_state(),
        )

        self.assertEqual(action, "LEFT")

    def test_exploration_uses_only_known_actions(self):
        agent = self.setup_agent(train=True)
        agent.epsilon = 1.0

        selected_actions = {
            callbacks.act(
                agent,
                game_state=self.make_game_state(),
            )
            for _ in range(100)
        }

        self.assertTrue(
            selected_actions.issubset(set(ACTIONS))
        )
        self.assertNotIn("BOMB", selected_actions)

    def test_exploration_ignores_wall_action(self):
        agent = self.setup_agent(train=True)
        agent.epsilon = 1.0

        game_state = self.make_game_state()
        self_x, self_y = game_state["self"][3]

        # Block LEFT.
        game_state["field"][self_x - 1, self_y] = -1

        selected_actions = {
            callbacks.act(
                agent,
                game_state=game_state,
            )
            for _ in range(200)
        }

        self.assertNotIn("LEFT", selected_actions)

    def test_greedy_action_ignores_wall_action(self):
        agent = self.setup_agent(train=False)

        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()

            # LEFT has the largest raw Q-value.
            agent.online_network.q_head[-1].bias[3] = 2.0

            # RIGHT is the best legal alternative.
            agent.online_network.q_head[-1].bias[1] = 1.0

        game_state = self.make_game_state()
        self_x, self_y = game_state["self"][3]

        # Block LEFT.
        game_state["field"][self_x - 1, self_y] = -1

        action = callbacks.act(
            agent,
            game_state,
        )

        self.assertEqual(action, "RIGHT")


if __name__ == "__main__":
    unittest.main()
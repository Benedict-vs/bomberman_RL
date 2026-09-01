import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_visit_count_increases_for_repeated_state(self):
        agent = self.setup_agent(train=False)
        game_state = self.make_game_state()
        game_state["round"] = 1
        game_state["step"] = 1

        with patch.object(callbacks, "VISIT_COUNT_ENABLED", True):
            first = callbacks._features_with_visit_count(agent, game_state)
            game_state["step"] = 2
            second = callbacks._features_with_visit_count(agent, game_state)

        self.assertAlmostEqual(first[8, 1, 1], 0.1)
        self.assertAlmostEqual(second[8, 1, 1], 0.2)

    def test_zero_control_keeps_visit_channel_zero(self):
        agent = self.setup_agent(train=False)
        game_state = self.make_game_state()
        game_state["round"] = 1
        game_state["step"] = 1

        with patch.object(callbacks, "VISIT_COUNT_ENABLED", False):
            first = callbacks._features_with_visit_count(agent, game_state)
            game_state["step"] = 2
            second = callbacks._features_with_visit_count(agent, game_state)

        self.assertEqual(float(first[8].sum()), 0.0)
        self.assertEqual(float(second[8].sum()), 0.0)

    def test_coin_reward_arm_uses_separate_model_and_source(self):
        environment = os.environ.copy()
        environment.update(
            {
                "BM_TASK2_ESCAPE_ARM": "reachable",
                "BM_TASK2_TRAINING_SEED": "11",
                "BM_TASK2_TOTAL_EPISODES": "2000",
                "BM_TASK2_FINETUNE_CRATE_WAIT": "1",
                "BM_TASK2_CRATE_WAIT_PENALTY": "-0.03",
                "BM_TASK2_FINETUNE_COIN_REWARD": "1",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent_code.ben_task2 import callbacks, train; "
                    "print(callbacks.MODEL_FILE); "
                    "print(callbacks.LOAD_MODEL_FILE); "
                    "print(train.EVENT_REWARDS['COIN_COLLECTED']); "
                    "print(train.POTENTIAL_REWARD_SCALE)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                (
                    "ben_task2_escape_crate_wait003_coin_reward15_"
                    "finetune2000_from_wait003_v1_seed11.pt"
                ),
                (
                    "ben_task2_escape_crate_wait003_"
                    "finetune2000_from10000_v1_seed11.pt"
                ),
                "1.5",
                "0.0",
            ],
        )

    def test_coin_reward_continuation_loads_a_new_output(self):
        environment = os.environ.copy()
        environment.update(
            {
                "BM_TASK2_ESCAPE_ARM": "reachable",
                "BM_TASK2_TRAINING_SEED": "11",
                "BM_TASK2_TOTAL_EPISODES": "5000",
                "BM_TASK2_FINETUNE_CRATE_WAIT": "1",
                "BM_TASK2_CRATE_WAIT_PENALTY": "-0.03",
                "BM_TASK2_FINETUNE_COIN_REWARD": "1",
                "BM_TASK2_CONTINUE_COIN_REWARD15": "1",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent_code.ben_task2 import callbacks, train; "
                    "print(callbacks.MODEL_FILE); "
                    "print(callbacks.LOAD_MODEL_FILE); "
                    "print(train.EVENT_REWARDS['COIN_COLLECTED']); "
                    "print(train.TOTAL_EPISODES)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                (
                    "ben_task2_escape_crate_wait003_coin_reward15_"
                    "continue5000_from_reward15_v1_seed11.pt"
                ),
                (
                    "ben_task2_escape_crate_wait003_coin_reward15_"
                    "finetune2000_from_wait003_v1_seed11.pt"
                ),
                "1.5",
                "5000",
            ],
        )


if __name__ == "__main__":
    unittest.main()

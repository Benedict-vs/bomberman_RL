import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as e
import numpy as np
import torch

from agent_code.ben_coin_collector_dqn import train
from agent_code.ben_coin_collector_dqn.model import CoinCollectorDQN


class TrainingCallbacksTest(unittest.TestCase):
    @staticmethod
    def make_game_state(
        round_number: int = 1,
        step: int = 1,
    ) -> dict:
        return {
            "round": round_number,
            "step": step,
            "field": np.zeros((17, 17), dtype=np.int8),
            "coins": [(5, 5)],
            "self": ("dqn-agent", 0, False, (1, 1)),
        }

    @staticmethod
    def make_agent() -> SimpleNamespace:
        return SimpleNamespace(
            online_network=CoinCollectorDQN(),
            logger=Mock(),
        )

    def test_reward_has_no_special_wait_penalty(self):
        movement_reward = train.reward_from_events(
            [e.MOVED_RIGHT]
        )
        waited_reward = train.reward_from_events(
            [e.WAITED]
        )

        self.assertAlmostEqual(
            movement_reward,
            train.STEP_REWARD,
        )
        self.assertAlmostEqual(
            waited_reward,
            train.STEP_REWARD,
        )
        self.assertEqual(movement_reward, waited_reward)

    def test_coin_and_invalid_action_rewards(self):
        coin_reward = train.reward_from_events(
            [e.COIN_COLLECTED]
        )
        invalid_reward = train.reward_from_events(
            [e.INVALID_ACTION]
        )

        self.assertAlmostEqual(coin_reward, 4.95)
        self.assertAlmostEqual(invalid_reward, -1.05)

    def test_epsilon_schedule(self):
        self.assertAlmostEqual(
            train._epsilon_for_step(0),
            train.EPSILON_START,
        )

        self.assertAlmostEqual(
            train._epsilon_for_step(
                train.EPSILON_DECAY_STEPS // 2
            ),
            0.525,
        )

        self.assertAlmostEqual(
            train._epsilon_for_step(
                train.EPSILON_DECAY_STEPS
            ),
            train.EPSILON_END,
        )

        self.assertAlmostEqual(
            train._epsilon_for_step(
                train.EPSILON_DECAY_STEPS * 2
            ),
            train.EPSILON_END,
        )

    @patch(
        "agent_code.ben_coin_collector_dqn.train."
        "torch.backends.mps.is_available",
        return_value=False,
    )
    @patch(
        "agent_code.ben_coin_collector_dqn.train.TrainLogger",
        None,
    )
    def test_setup_training_uses_cpu_fallback(
        self,
        mocked_mps_available,
    ):
        agent = self.make_agent()

        train.setup_training(agent)

        self.assertEqual(agent.device, torch.device("cpu"))
        self.assertEqual(agent.epsilon, train.EPSILON_START)
        self.assertEqual(len(agent.replay_buffer), 0)
        self.assertEqual(agent.environment_steps, 0)
        self.assertEqual(agent.optimization_steps, 0)
        self.assertIsNone(agent.trainlog)

        online_parameters = agent.online_network.state_dict()
        target_parameters = agent.target_network.state_dict()

        for name in online_parameters:
            self.assertTrue(
                torch.equal(
                    online_parameters[name],
                    target_parameters[name],
                )
            )

        for parameter in agent.target_network.parameters():
            self.assertFalse(parameter.requires_grad)

    @patch(
        "agent_code.ben_coin_collector_dqn.train."
        "torch.backends.mps.is_available",
        return_value=False,
    )
    @patch(
        "agent_code.ben_coin_collector_dqn.train.TrainLogger",
        None,
    )
    def test_end_of_round_marks_existing_transition_terminal(
        self,
        mocked_mps_available,
    ):
        agent = self.make_agent()
        train.setup_training(agent)

        old_game_state = self.make_game_state(
            round_number=1,
            step=1,
        )
        new_game_state = self.make_game_state(
            round_number=1,
            step=2,
        )

        train.game_events_occurred(
            agent,
            old_game_state=old_game_state,
            self_action="WAIT",
            new_game_state=new_game_state,
            events=[e.WAITED],
        )

        self.assertEqual(len(agent.replay_buffer), 1)
        self.assertEqual(agent.environment_steps, 1)

        previous_directory = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                os.chdir(temporary_directory)

                train.end_of_round(
                    agent,
                    last_game_state=old_game_state,
                    last_action="WAIT",
                    events=[e.WAITED, e.SURVIVED_ROUND],
                )

                self.assertTrue(
                    os.path.isfile(train.MODEL_FILE)
                )
        finally:
            os.chdir(previous_directory)

        # end_of_round must modify the existing transition,
        # not append a duplicate.
        self.assertEqual(len(agent.replay_buffer), 1)
        self.assertEqual(agent.environment_steps, 1)

        transition = agent.replay_buffer.sample(1)[0]

        self.assertEqual(transition.action, 4)
        self.assertAlmostEqual(
            transition.reward,
            train.STEP_REWARD,
        )
        self.assertTrue(transition.done)
        self.assertTrue(
            np.all(transition.next_state == 0.0)
        )

        self.assertEqual(agent.episode_reward, 0.0)
        self.assertEqual(agent.episode_events, [])
        self.assertEqual(agent.episode_losses, [])


if __name__ == "__main__":
    unittest.main()
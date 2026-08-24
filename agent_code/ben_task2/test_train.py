import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import events as e
import numpy as np
import torch

from agent_code.ben_task2 import train
from agent_code.ben_task2.model import (
    CoinCollectorDQN,
)


class TrainingCallbacksTest(unittest.TestCase):
    def test_task_2_artifact_names_do_not_reuse_task_1(self):
        self.assertEqual(
            train.MODEL_FILE,
            "ben_task2_safety_potential_v1_15000ep_seed11.pt",
        )
        self.assertEqual(
            train.RUN_LABEL,
            "task2_safety_potential_v1_15000ep_seed11",
        )
        self.assertEqual(
            train.TRAINLOG_OUT_DIR,
            "results/train/ben_task2",
        )
        self.assertEqual(
            train.CHECKPOINT_DIR,
            "../../results/train/ben_task2",
        )
        self.assertEqual(train.TRAINING_SEED, 11)

    @staticmethod
    def make_game_state(
        round_number: int = 1,
        step: int = 1,
    ) -> dict:
        return {
            "round": round_number,
            "step": step,
            "field": np.zeros(
                (17, 17),
                dtype=np.int8,
            ),
            "coins": [(5, 5)],
            "self": (
                "dqn-agent",
                0,
                False,
                (1, 1),
            ),
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
        self.assertEqual(
            movement_reward,
            waited_reward,
        )

    def test_coin_and_invalid_action_rewards(self):
        coin_reward = train.reward_from_events(
            [e.COIN_COLLECTED]
        )
        invalid_reward = train.reward_from_events(
            [e.INVALID_ACTION]
        )

        self.assertAlmostEqual(
            coin_reward,
            0.95,
        )
        self.assertAlmostEqual(
            invalid_reward,
            -1.05,
        )

    def test_destroying_crate_receives_task_2_reward(self):
        reward = train.reward_from_events(
            [e.CRATE_DESTROYED]
        )

        self.assertAlmostEqual(reward, 0.15)

    def test_death_penalty_is_applied_exactly_once(self):
        opponent_bomb_death = train.reward_from_events(
            [e.GOT_KILLED]
        )
        own_bomb_death = train.reward_from_events(
            [e.GOT_KILLED, e.KILLED_SELF]
        )

        self.assertAlmostEqual(opponent_bomb_death, -5.05)
        self.assertAlmostEqual(own_bomb_death, -5.05)

    def test_coin_potential_uses_shortest_walkable_distance(
        self,
    ):
        game_state = self.make_game_state()
        game_state["coins"] = [(3, 1)]
        game_state["self"] = (
            "dqn-agent",
            0,
            False,
            (1, 1),
        )

        # Block the direct route. The shortest legal path now has
        # four steps: up, right, right, down.
        game_state["field"][2, 1] = -1

        potential = train._coin_potential(
            game_state
        )

        self.assertAlmostEqual(
            potential,
            -4.0 / 32.0,
        )

    def test_potential_shaping_is_disabled_for_task_2_baseline(
        self,
    ):
        old_state = self.make_game_state()
        old_state["coins"] = [(4, 1)]
        old_state["self"] = (
            "dqn-agent",
            0,
            False,
            (2, 1),
        )

        closer_state = self.make_game_state()
        closer_state["coins"] = [(4, 1)]
        closer_state["self"] = (
            "dqn-agent",
            0,
            False,
            (3, 1),
        )

        farther_state = self.make_game_state()
        farther_state["coins"] = [(4, 1)]
        farther_state["self"] = (
            "dqn-agent",
            0,
            False,
            (1, 1),
        )

        closer_reward = train.potential_shaping_reward(
            old_state,
            closer_state,
        )
        farther_reward = train.potential_shaping_reward(
            old_state,
            farther_state,
        )

        self.assertEqual(closer_reward, 0.0)
        self.assertEqual(farther_reward, 0.0)

    def test_wait_remains_costly_with_safety_shaping(
        self,
    ):
        game_state = self.make_game_state()

        total_reward = (
            train.reward_from_events([e.WAITED])
            + train.potential_shaping_reward(
                game_state,
                game_state,
            )
            + train.safety_potential_shaping_reward(
                game_state,
                game_state,
            )
        )

        self.assertLess(
            total_reward,
            0.0,
        )

    def test_safety_shaping_rewards_leaving_bomb_danger(self):
        danger_state = self.make_game_state()
        danger_state["bombs"] = [((1, 1), 1)]

        safe_state = self.make_game_state()

        reward = train.safety_potential_shaping_reward(
            danger_state,
            safe_state,
        )

        self.assertAlmostEqual(reward, 0.79)

    def test_safety_shaping_penalizes_entering_bomb_danger(self):
        safe_state = self.make_game_state()

        danger_state = self.make_game_state()
        danger_state["bombs"] = [((1, 1), 1)]

        reward = train.safety_potential_shaping_reward(
            safe_state,
            danger_state,
        )

        self.assertAlmostEqual(reward, -0.802)

    def test_safety_shaping_penalizes_terminal_death(self):
        safe_state = self.make_game_state()

        reward = train.safety_potential_shaping_reward(
            safe_state,
            None,
        )

        self.assertAlmostEqual(reward, -1.0)

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
        "agent_code.ben_task2.train."
        "torch.backends.mps.is_available",
        return_value=False,
    )
    @patch(
        "agent_code.ben_task2.train.TrainLogger",
        None,
    )
    def test_setup_training_uses_cpu_fallback(
        self,
        mocked_mps_available,
    ):
        agent = self.make_agent()

        train.setup_training(agent)

        self.assertEqual(
            agent.device,
            torch.device("cpu"),
        )
        self.assertEqual(
            agent.epsilon,
            train.EPSILON_START,
        )
        self.assertEqual(
            len(agent.replay_buffer),
            0,
        )
        self.assertEqual(
            agent.environment_steps,
            0,
        )
        self.assertEqual(
            agent.optimization_steps,
            0,
        )
        self.assertIsNone(agent.trainlog)

        online_parameters = (
            agent.online_network.state_dict()
        )
        target_parameters = (
            agent.target_network.state_dict()
        )

        for name in online_parameters:
            self.assertTrue(
                torch.equal(
                    online_parameters[name],
                    target_parameters[name],
                )
            )

        for parameter in agent.target_network.parameters():
            self.assertFalse(
                parameter.requires_grad
            )

    @patch(
        "agent_code.ben_task2.train."
        "torch.backends.mps.is_available",
        return_value=False,
    )
    @patch(
        "agent_code.ben_task2.train.TrainLogger",
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

        self.assertEqual(
            len(agent.replay_buffer),
            1,
        )
        self.assertEqual(
            agent.environment_steps,
            1,
        )

        previous_directory = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                os.chdir(temporary_directory)

                train.end_of_round(
                    agent,
                    last_game_state=old_game_state,
                    last_action="WAIT",
                    events=[
                        e.WAITED,
                        e.SURVIVED_ROUND,
                    ],
                )

                self.assertTrue(
                    os.path.isfile(
                        train.MODEL_FILE
                    )
                )
        finally:
            os.chdir(previous_directory)

        # end_of_round must modify the existing transition,
        # not append a duplicate.
        self.assertEqual(
            len(agent.replay_buffer),
            1,
        )
        self.assertEqual(
            agent.environment_steps,
            1,
        )

        transition = agent.replay_buffer.sample(1)[0]

        expected_reward = (
            train.STEP_REWARD
            + train.potential_shaping_reward(
                old_game_state,
                new_game_state,
            )
            + train.safety_potential_shaping_reward(
                old_game_state,
                new_game_state,
            )
        )

        self.assertEqual(
            transition.action,
            5,
        )
        self.assertAlmostEqual(
            transition.reward,
            expected_reward,
        )
        self.assertTrue(
            transition.done
        )
        self.assertTrue(
            np.all(
                transition.next_state == 0.0
            )
        )

        self.assertEqual(
            agent.episode_reward,
            0.0,
        )
        self.assertEqual(
            agent.episode_events,
            [],
        )
        self.assertEqual(
            agent.episode_losses,
            [],
        )

    def _assert_terminal_death_transition(
        self,
        events: list[str],
    ) -> None:
        agent = self.make_agent()
        train.setup_training(agent)

        last_game_state = self.make_game_state()
        last_game_state["coins"] = []

        previous_directory = os.getcwd()

        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                os.chdir(temporary_directory)

                train.end_of_round(
                    agent,
                    last_game_state=last_game_state,
                    last_action="BOMB",
                    events=events,
                )
        finally:
            os.chdir(previous_directory)

        self.assertEqual(len(agent.replay_buffer), 1)
        self.assertEqual(agent.environment_steps, 1)

        transition = agent.replay_buffer.sample(1)[0]

        self.assertEqual(transition.action, 4)
        self.assertAlmostEqual(transition.reward, -6.05)
        self.assertTrue(transition.done)
        self.assertTrue(np.all(transition.next_state == 0.0))

    @patch(
        "agent_code.ben_task2.train."
        "torch.backends.mps.is_available",
        return_value=False,
    )
    @patch(
        "agent_code.ben_task2.train.TrainLogger",
        None,
    )
    def test_opponent_bomb_death_is_stored_as_terminal_transition(
        self,
        mocked_mps_available,
    ):
        self._assert_terminal_death_transition(
            [e.GOT_KILLED]
        )

    @patch(
        "agent_code.ben_task2.train."
        "torch.backends.mps.is_available",
        return_value=False,
    )
    @patch(
        "agent_code.ben_task2.train.TrainLogger",
        None,
    )
    def test_suicide_penalty_is_not_applied_twice(
        self,
        mocked_mps_available,
    ):
        self._assert_terminal_death_transition(
            [e.KILLED_SELF, e.GOT_KILLED]
        )


if __name__ == "__main__":
    unittest.main()

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
            "ben_task2_escape_reachable_tiles_v1_7000ep_seed11.pt",
        )
        self.assertEqual(
            train.RUN_LABEL,
            "task2_escape_reachable_tiles_v1_7000ep_seed11",
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
        self.assertEqual(train.TOTAL_EPISODES, 7000)
        self.assertEqual(
            train.ESCAPE_FEATURE_MODE,
            "reachable_safe_tiles",
        )
        self.assertIsNone(train.MULTISEED_ARM)
        self.assertFalse(train.FINE_TUNE_COIN_POTENTIAL)
        self.assertFalse(train.FINE_TUNE_COIN_REWARD)
        self.assertEqual(train.POTENTIAL_REWARD_SCALE, 0.0)

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
            online_network=CoinCollectorDQN(input_channels=10),
            logger=Mock(),
        )

    def test_wait_has_no_additional_penalty(self):
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
        self.assertEqual(waited_reward, movement_reward)

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

        self.assertAlmostEqual(reward, 0.25)

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

    def test_coin_potential_rewards_approach_and_penalizes_retreat(
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

        with patch.object(train, "POTENTIAL_REWARD_SCALE", 1.0):
            closer_reward = train.potential_shaping_reward(
                old_state,
                closer_state,
            )
            farther_reward = train.potential_shaping_reward(
                old_state,
                farther_state,
            )

        self.assertAlmostEqual(closer_reward, 1.01 / 32.0)
        self.assertAlmostEqual(farther_reward, -0.97 / 32.0)

    def test_coin_potential_ignores_crate_without_visible_coin(self):
        old_state = self.make_game_state()
        old_state["coins"] = []
        old_state["field"][3, 1] = 1
        old_state["self"] = ("dqn-agent", 0, False, (1, 1))

        closer_state = self.make_game_state()
        closer_state["coins"] = []
        closer_state["field"][3, 1] = 1
        closer_state["self"] = ("dqn-agent", 0, False, (2, 1))

        reward = train.potential_shaping_reward(old_state, closer_state)

        self.assertEqual(reward, 0.0)

    def test_navigation_potential_prioritizes_reachable_visible_coin(self):
        game_state = self.make_game_state()
        game_state["coins"] = [(5, 1)]
        game_state["field"][2, 1] = 1
        game_state["self"] = ("dqn-agent", 0, False, (1, 1))

        self.assertAlmostEqual(
            train._navigation_potential(game_state),
            -6.0 / 32.0,
        )

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

    def test_safe_coin_wait_penalty_is_disabled_for_crate_reward_arm(self):
        game_state = self.make_game_state()

        self.assertEqual(
            train.safe_coin_wait_penalty(game_state, "WAIT"),
            0.0,
        )

    def test_conditional_wait_penalty_preserves_dangerous_wait(self):
        game_state = self.make_game_state()
        game_state["bombs"] = [((1, 1), 1)]

        self.assertEqual(
            train.safe_coin_wait_penalty(game_state, "WAIT"),
            0.0,
        )

    def test_conditional_wait_penalty_requires_wait_and_coin(self):
        game_state = self.make_game_state()
        self.assertEqual(
            train.safe_coin_wait_penalty(game_state, "RIGHT"),
            0.0,
        )

        game_state["coins"] = []
        self.assertEqual(
            train.safe_coin_wait_penalty(game_state, "WAIT"),
            0.0,
        )

    def test_safe_crate_wait_penalty_requires_productive_safe_bomb(self):
        game_state = self.make_game_state()
        game_state["self"] = ("dqn-agent", 0, True, (1, 1))
        game_state["field"][2, 1] = 1

        with patch.object(train, "SAFE_CRATE_WAIT_PENALTY", -0.02):
            self.assertAlmostEqual(
                train.safe_crate_wait_penalty(game_state, "WAIT"),
                -0.02,
            )
            game_state["bombs"] = [((1, 1), 1)]
            self.assertEqual(
                train.safe_crate_wait_penalty(game_state, "WAIT"),
                0.0,
            )

    @patch(
        "agent_code.ben_task2.train.torch.backends.mps.is_available",
        return_value=False,
    )
    @patch("agent_code.ben_task2.train.TrainLogger", None)
    def test_replay_transition_advances_visit_count(
        self,
        mocked_mps_available,
    ):
        agent = self.make_agent()
        train.setup_training(agent)
        old_state = self.make_game_state(step=1)
        new_state = self.make_game_state(step=2)

        agent.visit_counts = np.zeros_like(
            old_state["field"],
            dtype=np.float32,
        )
        agent.visit_counts[1, 1] = 1.0
        augmented_old_state = dict(old_state)
        augmented_old_state["visit_counts"] = agent.visit_counts
        augmented_old_state["visit_count_encoding"] = "linear_10"
        augmented_old_state["escape_feature_mode"] = (
            train.ESCAPE_FEATURE_MODE
        )
        agent.last_action_features = train.state_to_features(
            augmented_old_state
        )
        agent.last_feature_round_step = (1, 1)

        with patch.object(train, "VISIT_COUNT_ENABLED", True):
            train.game_events_occurred(
                agent,
                old_game_state=old_state,
                self_action="WAIT",
                new_game_state=new_state,
                events=[e.WAITED],
            )

        transition = agent.replay_buffer.sample(1)[0]
        self.assertEqual(transition.state.shape, (10, 17, 17))
        self.assertEqual(transition.next_state.shape, (10, 17, 17))
        self.assertAlmostEqual(
            transition.state[8, 1, 1],
            0.1,
        )
        self.assertAlmostEqual(
            transition.next_state[8, 1, 1],
            0.2,
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
            train.reward_from_events([e.WAITED])
            + train.potential_shaping_reward(
                old_game_state,
                new_game_state,
            )
            + train.safety_potential_shaping_reward(
                old_game_state,
                new_game_state,
            )
            + train.safe_coin_wait_penalty(
                old_game_state,
                "WAIT",
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
        expected_reward = (
            train.reward_from_events(events)
            + train.potential_shaping_reward(last_game_state, None)
            + train.safety_potential_shaping_reward(last_game_state, None)
            + train.safe_coin_wait_penalty(last_game_state, "BOMB")
        )

        self.assertEqual(transition.action, 4)
        self.assertAlmostEqual(transition.reward, expected_reward)
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

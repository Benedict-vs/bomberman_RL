import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from agent_code.ben_task4 import train
from agent_code.ben_task4.replay_buffer import ReplayBuffer, Transition


class NStepReturnTest(unittest.TestCase):
    def setUp(self):
        self.previous_n = train.N_STEP_RETURN
        train.N_STEP_RETURN = 3
        self.agent = SimpleNamespace(
            n_step_queue=deque(),
            replay_buffer=ReplayBuffer(20),
        )

    def tearDown(self):
        train.N_STEP_RETURN = self.previous_n

    def append(self, value, reward, done=False):
        state = np.full((11, 2, 2), value, dtype=np.float32)
        next_state = None if done else state + 1
        train._append_n_step_transition(
            self.agent, state, 0, reward, next_state, done
        )

    def test_three_steps_use_discounted_reward_and_three_step_bootstrap(self):
        self.append(0, 1.0)
        self.append(1, 2.0)
        self.append(2, 3.0)
        self.assertEqual(len(self.agent.replay_buffer), 0)
        self.append(3, 4.0)
        transition = self.agent.replay_buffer._transitions[0]
        self.assertAlmostEqual(
            transition.reward,
            1.0 + train.GAMMA * 2.0 + train.GAMMA**2 * 3.0,
        )
        self.assertEqual(transition.n_steps, 3)
        self.assertFalse(transition.done)
        np.testing.assert_array_equal(
            transition.next_state,
            np.full((11, 2, 2), 3, dtype=np.float32),
        )

    def test_terminal_step_flushes_short_returns(self):
        self.append(0, 1.0)
        self.append(1, 2.0, done=True)
        transitions = list(self.agent.replay_buffer._transitions)
        self.assertEqual([t.n_steps for t in transitions], [2, 1])
        self.assertTrue(all(t.done for t in transitions))
        self.assertEqual(len(self.agent.n_step_queue), 0)
        self.assertAlmostEqual(transitions[0].reward, 1.0 + train.GAMMA * 2.0)
        self.assertAlmostEqual(transitions[1].reward, 2.0)

    def test_survival_marks_pending_last_step_terminal_before_flush(self):
        self.append(0, 1.0)
        self.append(1, 2.0)
        train._finish_n_step_episode(self.agent)
        transitions = list(self.agent.replay_buffer._transitions)
        self.assertEqual([t.n_steps for t in transitions], [2, 1])
        self.assertTrue(all(t.done for t in transitions))

    def test_augmentation_preserves_n_steps(self):
        from agent_code.ben_task4.augmentation import (
            all_symmetry_transforms,
            transform_transition,
        )

        transition = Transition(
            state=np.zeros((11, 2, 2), dtype=np.float32),
            action=4,
            reward=1.0,
            next_state=np.zeros((11, 2, 2), dtype=np.float32),
            done=False,
            n_steps=3,
        )
        self.assertEqual(
            transform_transition(transition, rotations=1, reflect=True).n_steps,
            3,
        )
        transforms = all_symmetry_transforms(transition)
        self.assertEqual(len(transforms), 8)
        self.assertTrue(all(item.action == 4 for item in transforms))
        self.assertTrue(all(item.n_steps == 3 for item in transforms))

    def test_one_step_control_stores_nonterminal_transition_immediately(self):
        train.N_STEP_RETURN = 1
        self.append(0, 1.0)
        self.assertEqual(len(self.agent.replay_buffer), 1)
        self.assertEqual(len(self.agent.n_step_queue), 0)
        transition = self.agent.replay_buffer._transitions[0]
        self.assertEqual(transition.n_steps, 1)
        self.assertFalse(transition.done)


if __name__ == "__main__":
    unittest.main()

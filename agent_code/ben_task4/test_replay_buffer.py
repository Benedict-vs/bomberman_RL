import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from agent_code.ben_task4 import train
from agent_code.ben_task4.replay_buffer import BombOutcomeBuffer, ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def transition_state(self, value: int) -> np.ndarray:
        return np.full((11, 2, 2), value, dtype=np.float32)

    def test_uniform_buffer_remains_uniform_mode(self):
        buffer = ReplayBuffer(20)
        buffer.append(self.transition_state(0), 0, 0.0, None, False)
        self.assertFalse(buffer.event_balance)
        self.assertEqual(len(buffer.sample(1)), 1)

    def test_balanced_buffer_samples_rare_event_buckets(self):
        buffer = ReplayBuffer(40, event_balance=True)
        for value in range(35):
            buffer.append(
                self.transition_state(value),
                0,
                0.0,
                None,
                False,
                event_tags=(frozenset({"kill"}) if value == 0 else frozenset()),
            )
        for value in range(35, 40):
            buffer.append(
                self.transition_state(value),
                0,
                0.0,
                None,
                False,
                event_tags=frozenset({"death"}),
            )

        sample = buffer.sample(20)
        tags = [tag for transition in sample for tag in transition.event_tags]
        self.assertIn("kill", tags)
        self.assertIn("death", tags)
        self.assertEqual(len(sample), 20)

    def test_prioritized_buffer_updates_td_error_priorities(self):
        buffer = ReplayBuffer(4, prioritized=True)
        for value in range(4):
            buffer.append(self.transition_state(value), 0, 0.0, None, False)
        buffer.update_priorities([0, 1], np.array([0.1, 9.0]))
        np.random.seed(7)
        _sample, indices, weights = buffer.sample_prioritized(20, beta=0.4)
        self.assertGreater(indices.count(1), indices.count(0))
        self.assertEqual(weights.shape, (20,))
        self.assertLessEqual(float(weights.max()), 1.0)

    def test_marking_terminal_can_correct_stored_reward(self):
        buffer = ReplayBuffer(4)
        buffer.append(self.transition_state(1), 0, 2.5, self.transition_state(2), False)

        buffer.mark_last_terminal(reward_adjustment=-0.99)

        transition = buffer._transitions[-1]
        self.assertTrue(transition.done)
        self.assertAlmostEqual(transition.reward, 1.51)
        self.assertTrue(np.all(transition.next_state == 0.0))

    def test_terminal_safety_adjustment_isolated_to_candidate(self):
        with patch.object(train, "TERMINAL_SAFETY_POTENTIAL_ZERO", False):
            self.assertEqual(
                train._terminal_safety_shaping_adjustment({}),
                0.0,
            )
        with (
            patch.object(train, "TERMINAL_SAFETY_POTENTIAL_ZERO", True),
            patch.object(train, "_safety_potential", return_value=0.8),
        ):
            self.assertAlmostEqual(
                train._terminal_safety_shaping_adjustment({}),
                -train.GAMMA * 0.8,
            )

    def test_bomb_outcome_buffer_accepts_only_resolved_classes(self):
        buffer = BombOutcomeBuffer(2)
        state = self.transition_state(3)
        buffer.append(state, 1)
        state.fill(0.0)
        example = buffer.sample(1)[0]
        self.assertEqual(example.outcome, 1)
        self.assertTrue(np.all(example.state == 3.0))
        with self.assertRaises(ValueError):
            buffer.append(self.transition_state(0), 3)

    def test_pending_bomb_resolves_only_after_disappearing(self):
        buffer = BombOutcomeBuffer()
        agent = SimpleNamespace(
            bomb_outcome_buffer=buffer,
            pending_bomb_outcome={
                "position": (3, 3), "state": self.transition_state(4),
                "kill": False, "self_death": False,
            },
        )
        train._observe_pending_bomb(agent, {"bombs": [((3, 3), 1)]}, [])
        self.assertEqual(len(buffer), 0)
        train._observe_pending_bomb(agent, {"bombs": []}, ["KILLED_OPPONENT"])
        example = buffer.sample(1)[0]
        self.assertEqual(example.outcome, 1)
        self.assertIsNone(agent.pending_bomb_outcome)


if __name__ == "__main__":
    unittest.main()

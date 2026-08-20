import unittest

import numpy as np

from replay_buffer import ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    @staticmethod
    def make_state(value: float) -> np.ndarray:
        return np.full(
            (3, 17, 17),
            value,
            dtype=np.float32,
        )

    def test_append_and_sample(self):
        buffer = ReplayBuffer(capacity=10)

        buffer.append(
            state=self.make_state(1.0),
            action=2,
            reward=0.5,
            next_state=self.make_state(2.0),
            done=False,
        )

        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.capacity, 10)

        transition = buffer.sample(1)[0]

        self.assertTrue(np.all(transition.state == 1.0))
        self.assertEqual(transition.action, 2)
        self.assertEqual(transition.reward, 0.5)
        self.assertTrue(np.all(transition.next_state == 2.0))
        self.assertFalse(transition.done)

    def test_capacity_removes_oldest_transition(self):
        buffer = ReplayBuffer(capacity=2)

        for action in range(3):
            buffer.append(
                state=self.make_state(float(action)),
                action=action,
                reward=0.0,
                next_state=self.make_state(float(action + 1)),
                done=False,
            )

        self.assertEqual(len(buffer), 2)

        remaining_actions = {
            transition.action
            for transition in buffer.sample(2)
        }

        self.assertEqual(remaining_actions, {1, 2})

    def test_stored_states_are_copies(self):
        buffer = ReplayBuffer(capacity=1)

        state = self.make_state(1.0)
        next_state = self.make_state(2.0)

        buffer.append(
            state=state,
            action=0,
            reward=0.0,
            next_state=next_state,
            done=False,
        )

        state.fill(99.0)
        next_state.fill(99.0)

        transition = buffer.sample(1)[0]

        self.assertTrue(np.all(transition.state == 1.0))
        self.assertTrue(np.all(transition.next_state == 2.0))

    def test_terminal_transition_gets_zero_next_state(self):
        buffer = ReplayBuffer(capacity=1)
        state = self.make_state(1.0)

        buffer.append(
            state=state,
            action=4,
            reward=-0.05,
            next_state=None,
            done=True,
        )

        transition = buffer.sample(1)[0]

        self.assertTrue(np.all(transition.next_state == 0.0))
        self.assertTrue(transition.done)

    def test_mark_last_transition_terminal(self):
        buffer = ReplayBuffer(capacity=2)

        buffer.append(
            state=self.make_state(1.0),
            action=3,
            reward=-0.05,
            next_state=self.make_state(2.0),
            done=False,
        )

        buffer.mark_last_terminal()
        transition = buffer.sample(1)[0]

        self.assertEqual(len(buffer), 1)
        self.assertEqual(transition.action, 3)
        self.assertAlmostEqual(transition.reward, -0.05)
        self.assertTrue(transition.done)
        self.assertTrue(
            np.all(transition.next_state == 0.0)
        )

    def test_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0)

        buffer = ReplayBuffer(capacity=2)

        with self.assertRaises(ValueError):
            buffer.sample(0)

        with self.assertRaises(ValueError):
            buffer.sample(1)


if __name__ == "__main__":
    unittest.main()
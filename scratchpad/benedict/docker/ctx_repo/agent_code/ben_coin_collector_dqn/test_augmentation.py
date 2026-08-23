import unittest

import numpy as np

from agent_code.ben_coin_collector_dqn.augmentation import (
    transform_transition,
)
from agent_code.ben_coin_collector_dqn.model import ACTIONS
from agent_code.ben_coin_collector_dqn.replay_buffer import Transition


class SymmetryAugmentationTest(unittest.TestCase):
    @staticmethod
    def make_transition(
        action: str,
        reward: float = 1.5,
        done: bool = False,
    ) -> Transition:
        state = np.zeros(
            (3, 5, 5),
            dtype=np.float32,
        )
        next_state = np.zeros_like(state)

        # Agent above the center.
        state[2, 1, 2] = 1.0

        # Agent above and right of the center.
        next_state[2, 1, 3] = 1.0

        return Transition(
            state=state,
            action=ACTIONS.index(action),
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def test_counterclockwise_rotation_transforms_action(self):
        transition = self.make_transition(
            action="UP"
        )

        transformed = transform_transition(
            transition,
            rotations=1,
            reflect=False,
        )

        # UP becomes LEFT after a 90-degree
        # counterclockwise rotation.
        self.assertEqual(
            ACTIONS[transformed.action],
            "LEFT",
        )

        # The agent position is rotated with the board.
        self.assertEqual(
            transformed.state[2, 2, 1],
            1.0,
        )

        self.assertEqual(
            transformed.next_state[2, 1, 1],
            1.0,
        )

    def test_horizontal_reflection_transforms_action(self):
        transition = self.make_transition(
            action="RIGHT"
        )

        transformed = transform_transition(
            transition,
            rotations=0,
            reflect=True,
        )

        self.assertEqual(
            ACTIONS[transformed.action],
            "LEFT",
        )

        # Horizontal reflection changes x but not y.
        self.assertEqual(
            transformed.state[2, 1, 2],
            1.0,
        )

        self.assertEqual(
            transformed.next_state[2, 1, 1],
            1.0,
        )

    def test_wait_is_unchanged_by_every_symmetry(self):
        transition = self.make_transition(
            action="WAIT"
        )

        for rotations in range(4):
            for reflect in (False, True):
                transformed = transform_transition(
                    transition,
                    rotations=rotations,
                    reflect=reflect,
                )

                self.assertEqual(
                    ACTIONS[transformed.action],
                    "WAIT",
                )

    def test_four_rotations_restore_transition(self):
        transition = self.make_transition(
            action="DOWN"
        )

        transformed = transform_transition(
            transition,
            rotations=4,
            reflect=False,
        )

        self.assertEqual(
            transformed.action,
            transition.action,
        )
        self.assertTrue(
            np.array_equal(
                transformed.state,
                transition.state,
            )
        )
        self.assertTrue(
            np.array_equal(
                transformed.next_state,
                transition.next_state,
            )
        )

    def test_reward_done_and_memory_layout_are_preserved(self):
        transition = self.make_transition(
            action="RIGHT",
            reward=-0.75,
            done=True,
        )

        transformed = transform_transition(
            transition,
            rotations=3,
            reflect=True,
        )

        self.assertAlmostEqual(
            transformed.reward,
            -0.75,
        )
        self.assertTrue(transformed.done)
        self.assertTrue(
            transformed.state.flags.c_contiguous
        )
        self.assertTrue(
            transformed.next_state.flags.c_contiguous
        )


if __name__ == "__main__":
    unittest.main()
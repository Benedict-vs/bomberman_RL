import unittest

import torch

from agent_code.ben_coin_collector_dqn.action_mask import (
    legal_action_mask,
)
from agent_code.ben_coin_collector_dqn.features import (
    SELF_CHANNEL,
    WALL_CHANNEL,
)


class LegalActionMaskTest(unittest.TestCase):
    def test_blocks_walls(self):
        states = torch.zeros(
            (1, 3, 17, 17),
            dtype=torch.float32,
        )

        states[0, SELF_CHANNEL, 5, 5] = 1.0

        # Walls above and left.
        states[0, WALL_CHANNEL, 4, 5] = 1.0
        states[0, WALL_CHANNEL, 5, 4] = 1.0

        mask = legal_action_mask(states)

        expected = torch.tensor(
            [[False, True, True, False, True]]
        )

        self.assertTrue(torch.equal(mask, expected))

    def test_blocks_board_boundaries(self):
        states = torch.zeros(
            (1, 3, 17, 17),
            dtype=torch.float32,
        )

        states[0, SELF_CHANNEL, 0, 0] = 1.0

        mask = legal_action_mask(states)

        expected = torch.tensor(
            [[False, True, True, False, True]]
        )

        self.assertTrue(torch.equal(mask, expected))

    def test_supports_batches_and_keeps_wait_legal(self):
        states = torch.zeros(
            (2, 3, 17, 17),
            dtype=torch.float32,
        )

        states[0, SELF_CHANNEL, 5, 5] = 1.0
        states[1, SELF_CHANNEL, 8, 8] = 1.0

        mask = legal_action_mask(states)

        self.assertEqual(mask.shape, (2, 5))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(mask[:, 4].all())

    def test_rejects_unbatched_state(self):
        state = torch.zeros(
            (3, 17, 17),
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            legal_action_mask(state)


if __name__ == "__main__":
    unittest.main()
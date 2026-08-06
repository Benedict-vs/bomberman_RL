import unittest

import numpy as np

from features import (
    COIN_CHANNEL,
    N_CHANNELS,
    SELF_CHANNEL,
    WALL_CHANNEL,
    state_to_features,
)


class StateToFeaturesTest(unittest.TestCase):
    def test_encodes_coin_heaven_state(self):
        # Framework convention: field[x, y].
        # A non-square field makes transpose errors visible.
        field = np.zeros((4, 3), dtype=np.int8)
        field[0, 1] = -1
        field[3, 2] = -1

        game_state = {
            "field": field,
            "coins": [(3, 1), (1, 2)],
            "self": ("dqn-agent", 0, False, (2, 2)),
        }

        features = state_to_features(game_state)

        self.assertEqual(features.shape, (N_CHANNELS, 3, 4))
        self.assertEqual(features.dtype, np.float32)

        self.assertEqual(features[WALL_CHANNEL, 1, 0], 1.0)
        self.assertEqual(features[WALL_CHANNEL, 2, 3], 1.0)

        self.assertEqual(features[COIN_CHANNEL, 1, 3], 1.0)
        self.assertEqual(features[COIN_CHANNEL, 2, 1], 1.0)

        self.assertEqual(features[SELF_CHANNEL, 2, 2], 1.0)

        self.assertEqual(features[WALL_CHANNEL].sum(), 2.0)
        self.assertEqual(features[COIN_CHANNEL].sum(), 2.0)
        self.assertEqual(features[SELF_CHANNEL].sum(), 1.0)

    def test_returns_none_for_missing_state(self):
        self.assertIsNone(state_to_features(None))


if __name__ == "__main__":
    unittest.main()
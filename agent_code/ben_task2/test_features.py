import unittest

import numpy as np

from agent_code.ben_task2.features import (
    BOMB_AVAILABLE_CHANNEL,
    BOMB_TIMER_CHANNEL,
    COIN_CHANNEL,
    CRATE_CHANNEL,
    DANGER_CHANNEL,
    EXPLOSION_CHANNEL,
    N_CHANNELS,
    SELF_CHANNEL,
    WALL_CHANNEL,
    legal_action_mask,
    state_to_features,
)


class StateToFeaturesTest(unittest.TestCase):
    def test_encodes_task_2_state(self):
        # Framework convention: field[x, y].
        # A non-square field makes transpose errors visible.
        field = np.zeros((4, 3), dtype=np.int8)
        field[0, 1] = -1
        field[3, 2] = -1
        field[1, 1] = 1

        explosion_map = np.zeros_like(field)
        explosion_map[2, 0] = 1

        game_state = {
            "field": field,
            "coins": [(3, 1), (1, 2)],
            "self": ("dqn-agent", 0, True, (2, 2)),
            "bombs": [((2, 1), 3)],
            "explosion_map": explosion_map,
        }

        features = state_to_features(game_state)

        self.assertEqual(features.shape, (N_CHANNELS, 3, 4))
        self.assertEqual(features.dtype, np.float32)

        self.assertEqual(features[WALL_CHANNEL, 1, 0], 1.0)
        self.assertEqual(features[WALL_CHANNEL, 2, 3], 1.0)
        self.assertEqual(features[CRATE_CHANNEL, 1, 1], 1.0)

        self.assertEqual(features[COIN_CHANNEL, 1, 3], 1.0)
        self.assertEqual(features[COIN_CHANNEL, 2, 1], 1.0)

        self.assertEqual(features[SELF_CHANNEL, 2, 2], 1.0)
        self.assertAlmostEqual(features[BOMB_TIMER_CHANNEL, 1, 2], 0.4)
        self.assertEqual(features[EXPLOSION_CHANNEL, 0, 2], 1.0)
        self.assertTrue(np.all(features[BOMB_AVAILABLE_CHANNEL] == 1.0))

        self.assertEqual(features[WALL_CHANNEL].sum(), 2.0)
        self.assertEqual(features[COIN_CHANNEL].sum(), 2.0)
        self.assertEqual(features[SELF_CHANNEL].sum(), 1.0)

    def test_danger_uses_blast_geometry_and_earliest_timer(self):
        field = np.zeros((9, 9), dtype=np.int8)
        field[6, 4] = -1
        field[4, 5] = 1

        game_state = {
            "field": field,
            "coins": [],
            "self": ("dqn-agent", 0, False, (1, 1)),
            "bombs": [((4, 4), 4), ((2, 4), 1)],
            "explosion_map": np.zeros_like(field),
        }

        danger = state_to_features(game_state)[DANGER_CHANNEL]

        self.assertAlmostEqual(danger[4, 4], 0.8)
        self.assertAlmostEqual(danger[4, 3], 0.8)
        self.assertAlmostEqual(danger[4, 5], 0.8)
        self.assertEqual(danger[4, 6], 0.0)
        self.assertAlmostEqual(danger[6, 4], 0.2)
        self.assertAlmostEqual(danger[7, 4], 0.2)
        self.assertEqual(danger[8, 4], 0.0)

    def test_active_explosion_is_immediate_danger(self):
        field = np.zeros((3, 3), dtype=np.int8)
        explosion_map = np.zeros_like(field)
        explosion_map[2, 1] = 1

        game_state = {
            "field": field,
            "coins": [],
            "self": ("dqn-agent", 0, False, (0, 0)),
            "bombs": [],
            "explosion_map": explosion_map,
        }

        features = state_to_features(game_state)

        self.assertEqual(features[EXPLOSION_CHANNEL, 1, 2], 1.0)
        self.assertEqual(features[DANGER_CHANNEL, 1, 2], 1.0)
        self.assertTrue(np.all(features[BOMB_AVAILABLE_CHANNEL] == 0.0))

    def test_legal_action_mask_excludes_blocked_actions(self):
        field = np.zeros((5, 5), dtype=np.int8)
        field[1, 0] = -1
        field[2, 1] = 1

        game_state = {
            "field": field,
            "coins": [],
            "self": ("dqn-agent", 0, False, (1, 1)),
            "bombs": [((1, 2), 3)],
            "explosion_map": np.zeros_like(field),
        }

        mask = legal_action_mask(state_to_features(game_state))

        np.testing.assert_array_equal(
            mask,
            [False, False, False, True, False, True],
        )

    def test_legal_action_mask_allows_available_bomb(self):
        field = np.zeros((3, 3), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("dqn-agent", 0, True, (1, 1)),
            "bombs": [],
            "explosion_map": np.zeros_like(field),
        }

        mask = legal_action_mask(state_to_features(game_state))

        self.assertTrue(mask[4])
        self.assertTrue(mask[5])

    def test_returns_none_for_missing_state(self):
        self.assertIsNone(state_to_features(None))


if __name__ == "__main__":
    unittest.main()

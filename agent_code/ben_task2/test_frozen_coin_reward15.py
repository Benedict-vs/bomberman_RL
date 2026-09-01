import hashlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from agent_code.dqn_task2_coin_reward15 import callbacks


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task2"
    / (
        "ben_task2_escape_crate_wait003_coin_reward15_"
        "finetune2000_from_wait003_v1_seed11.pt"
    )
)
FROZEN_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "dqn_task2_coin_reward15"
    / "dqn_task2_coin_reward15_seed11.pt"
)


class FrozenCoinReward15Task2Test(unittest.TestCase):
    def test_model_is_byte_identical_to_audited_candidate(self):
        self.assertEqual(
            hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest(),
            hashlib.sha256(FROZEN_MODEL.read_bytes()).hexdigest(),
        )

    def test_frozen_agent_is_environment_independent(self):
        self.assertEqual(
            callbacks.MODEL_FILE,
            "dqn_task2_coin_reward15_seed11.pt",
        )
        self.assertEqual(callbacks.TRAINING_SEED, 11)
        self.assertEqual(
            callbacks.ESCAPE_FEATURE_MODE,
            "reachable_safe_tiles",
        )

    def test_frozen_agent_loads_on_cpu_and_acts_with_ten_channels(self):
        agent = SimpleNamespace(train=False, logger=Mock())
        previous_directory = os.getcwd()
        try:
            os.chdir(FROZEN_MODEL.parent)
            callbacks.setup(agent)
        finally:
            os.chdir(previous_directory)

        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "round": 1,
            "step": 1,
            "field": field,
            "coins": [(5, 5)],
            "self": ("dqn_task2_coin_reward15", 0, True, (1, 1)),
            "bombs": [],
            "explosion_map": np.zeros_like(field),
        }

        self.assertEqual(agent.device.type, "cpu")
        self.assertIn(callbacks.act(agent, game_state), callbacks.ACTIONS)
        self.assertEqual(agent.last_action_features.shape, (10, 17, 17))


if __name__ == "__main__":
    unittest.main()

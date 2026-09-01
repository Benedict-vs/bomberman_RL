import hashlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from agent_code.dqn_task3 import callbacks


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task3"
    / "ben_task3_mixed_safety_multiseed_v1_5000ep_seed13.pt"
)
FROZEN_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "dqn_task3"
    / "dqn_task3_seed13.pt"
)


class FrozenTask3Test(unittest.TestCase):
    def test_model_is_byte_identical_to_selected_seed13_candidate(self):
        self.assertEqual(
            hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest(),
            hashlib.sha256(FROZEN_MODEL.read_bytes()).hexdigest(),
        )

    def test_configuration_is_fixed_and_submission_local(self):
        self.assertEqual(callbacks.MODEL_FILE, "dqn_task3_seed13.pt")
        self.assertEqual(callbacks.INFERENCE_MODEL_FILE, callbacks.MODEL_FILE)
        self.assertEqual(callbacks.TRAINING_SEED, 13)
        self.assertEqual(
            callbacks.ESCAPE_FEATURE_MODE,
            "reachable_safe_tiles",
        )
        self.assertFalse(hasattr(callbacks, "MODEL_VARIANT"))
        self.assertFalse(hasattr(callbacks, "CHECKPOINT_EPISODE"))

    def test_agent_loads_on_cpu_and_acts_with_eleven_channels(self):
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
            "self": ("dqn_task3", 0, True, (1, 1)),
            "others": [("opponent", 0, False, (5, 1))],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
        }

        self.assertEqual(agent.device.type, "cpu")
        self.assertIn(callbacks.act(agent, game_state), callbacks.ACTIONS)
        self.assertEqual(agent.last_action_features.shape, (11, 17, 17))


if __name__ == "__main__":
    unittest.main()

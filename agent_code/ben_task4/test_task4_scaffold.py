import hashlib
import os
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from agent_code.ben_task4 import callbacks, train
from agent_code.ben_task4.features import (
    OPPONENT_BOMB_ALIGNMENT_CHANNEL,
    state_to_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK3_FROZEN_MODEL = (
    REPO_ROOT / "agent_code" / "dqn_task3" / "dqn_task3_seed13.pt"
)
TASK4_BASELINE_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task4"
    / "ben_task4_task3_baseline_seed13.pt"
)
TASK4_SAFE_11CH_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task4"
    / "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
)
TASK4_ALIGNMENT_12CH_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task4"
    / "ben_task4_safe_seed11_12ch_alignment_source.pt"
)


class Task4ScaffoldTest(unittest.TestCase):
    def test_local_baseline_is_byte_identical_to_frozen_task3(self):
        self.assertEqual(
            hashlib.sha256(TASK3_FROZEN_MODEL.read_bytes()).hexdigest(),
            hashlib.sha256(TASK4_BASELINE_MODEL.read_bytes()).hexdigest(),
        )

    def test_artifact_names_and_paths_are_task4_only(self):
        expected_sources = {
            "rule_based_continue_v1": (
                "ben_task4_baseline_v1_5000ep_seed11.pt"
            ),
            "mixed_kill_v1": "ben_task4_baseline_v1_5000ep_seed11.pt",
            "rule_based_continue_control1000_v1": (
                "ben_task4_rule_based_continue_v1_2000ep_seed11.pt"
            ),
            "mixed_kill_consolidate1000_v1": (
                "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
            ),
            "opponent_alignment_zero_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "opponent_alignment_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "opponent_alignment_mixed_zero_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "opponent_alignment_mixed_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
        }
        expected_source = expected_sources.get(
            callbacks.TRAINING_ARM,
            "ben_task4_task3_baseline_seed13.pt",
        )
        self.assertEqual(callbacks.LOAD_MODEL_FILE, expected_source)
        self.assertEqual(
            callbacks.MODEL_FILE,
            (
                f"ben_task4_{callbacks.EXPERIMENT_STEM}_"
                f"{callbacks.TOTAL_EPISODES}ep_seed11.pt"
            ),
        )
        self.assertEqual(
            train.RUN_LABEL,
            (
                f"task4_{callbacks.EXPERIMENT_STEM}_"
                f"{callbacks.TOTAL_EPISODES}ep_seed11"
            ),
        )
        self.assertEqual(train.TRAINLOG_OUT_DIR, "results/train/ben_task4")
        self.assertEqual(train.CHECKPOINT_DIR, "../../results/train/ben_task4")
        self.assertNotIn("ben_task3", callbacks.MODEL_FILE)
        self.assertNotIn("ben_task3", train.TRAINLOG_OUT_DIR)

    def test_inference_uses_local_baseline_on_cpu(self):
        agent = SimpleNamespace(train=False, logger=Mock())
        previous_directory = os.getcwd()
        try:
            os.chdir(TASK4_BASELINE_MODEL.parent)
            callbacks.setup(agent)
        finally:
            os.chdir(previous_directory)

        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "round": 1,
            "step": 1,
            "field": field,
            "coins": [(5, 5)],
            "self": ("ben_task4", 0, True, (1, 1)),
            "bombs": [],
            "others": [("rule-based", 0, False, (5, 1))],
            "explosion_map": np.zeros_like(field),
        }

        self.assertEqual(agent.device.type, "cpu")
        self.assertIn(callbacks.act(agent, game_state), callbacks.ACTIONS)
        self.assertEqual(
            agent.last_action_features.shape,
            (callbacks.INPUT_CHANNELS, 17, 17),
        )

    def test_training_output_does_not_reuse_baseline_name(self):
        self.assertNotEqual(callbacks.MODEL_FILE, callbacks.LOAD_MODEL_FILE)
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertFalse(
                (Path(temporary_directory) / callbacks.MODEL_FILE).exists()
            )

    def test_current_arm_keeps_rewards_and_records_lineup(self):
        import events as e

        self.assertEqual(train.EVENT_REWARDS[e.KILLED_OPPONENT], 5.0)
        self.assertEqual(train.EVENT_REWARDS[e.GOT_KILLED], -5.0)
        self.assertEqual(train.EVENT_REWARDS[e.KILLED_SELF], 0.0)
        expected_lineup = (
            "peaceful_agent,rule_based_agent,rule_based_agent"
            if callbacks.TRAINING_ARM
            in {
                "mixed_kill_v1",
                "opponent_alignment_mixed_zero_v1",
                "opponent_alignment_mixed_v1",
            }
            else "rule_based_agent,rule_based_agent,rule_based_agent"
        )
        self.assertEqual(train.TRAINING_OPPONENTS, expected_lineup)

    def test_consolidation_pair_uses_expected_sources_and_same_setup(self):
        expected_sources = {
            "rule_based_continue_control1000_v1": (
                "ben_task4_rule_based_continue_v1_2000ep_seed11.pt"
            ),
            "mixed_kill_consolidate1000_v1": (
                "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
            ),
        }
        outputs = {}
        for arm, expected_source in expected_sources.items():
            environment = os.environ.copy()
            environment["BM_TASK4_TRAINING_ARM"] = arm
            environment["BM_TASK4_TOTAL_EPISODES"] = "1000"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import events as e; "
                        "from agent_code.ben_task4 import callbacks, train; "
                        "print(callbacks.LOAD_MODEL_FILE); "
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                        "print(train.EVENT_REWARDS[e.KILLED_SELF]); "
                        "print(train.TRAINING_OPPONENTS)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs[arm] = result.stdout.splitlines()
            self.assertEqual(outputs[arm][0], expected_source)

        values = list(outputs.values())
        self.assertEqual(values[0][1:], values[1][1:])

    def test_alignment_mixed_pair_differs_only_in_channel_mode(self):
        outputs = {}
        for arm in (
            "opponent_alignment_mixed_zero_v1",
            "opponent_alignment_mixed_v1",
        ):
            environment = os.environ.copy()
            environment["BM_TASK4_TRAINING_ARM"] = arm
            environment["BM_TASK4_TOTAL_EPISODES"] = "1000"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import events as e; "
                        "from agent_code.ben_task4 import callbacks, train; "
                        "print(callbacks.LOAD_MODEL_FILE); "
                        "print(callbacks.INPUT_CHANNELS); "
                        "print(callbacks.OPPONENT_ALIGNMENT_MODE); "
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                        "print(train.EVENT_REWARDS[e.KILLED_SELF]); "
                        "print(train.TRAINING_OPPONENTS)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs[arm] = result.stdout.splitlines()

        zero = outputs["opponent_alignment_mixed_zero_v1"]
        enabled = outputs["opponent_alignment_mixed_v1"]
        self.assertEqual(zero[0:2], enabled[0:2])
        self.assertEqual(zero[3:], enabled[3:])
        self.assertEqual(zero[2], "zero")
        self.assertEqual(enabled[2], "enabled")

    def test_alignment_source_preserves_old_channels_and_zeros_new_one(self):
        source = __import__("torch").load(
            TASK4_SAFE_11CH_MODEL,
            map_location="cpu",
            weights_only=True,
        )
        converted = __import__("torch").load(
            TASK4_ALIGNMENT_12CH_MODEL,
            map_location="cpu",
            weights_only=True,
        )
        first_weight = "convolutional.0.weight"
        self.assertEqual(converted[first_weight].shape[1], 12)
        self.assertTrue(
            __import__("torch").equal(
                converted[first_weight][:, :11],
                source[first_weight],
            )
        )
        self.assertEqual(float(converted[first_weight][:, 11].abs().sum()), 0.0)
        for key in source:
            if key != first_weight:
                self.assertTrue(__import__("torch").equal(converted[key], source[key]))

    def test_alignment_channel_marks_bomb_lines_without_prescribing_action(self):
        field = np.zeros((17, 17), dtype=np.int8)
        field[6, 5] = -1
        field[4, 5] = 1
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task4", 0, True, (1, 1)),
            "others": [("opponent", 0, True, (5, 5))],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
            "opponent_alignment_mode": "enabled",
        }
        features = state_to_features(game_state)
        alignment = features[OPPONENT_BOMB_ALIGNMENT_CHANNEL]
        self.assertEqual(features.shape, (12, 17, 17))
        self.assertEqual(float(alignment[5, 2]), 1.0)
        self.assertEqual(float(alignment[2, 5]), 1.0)
        self.assertEqual(float(alignment[5, 7]), 0.0)

        game_state["opponent_alignment_mode"] = "zero"
        zero_features = state_to_features(game_state)
        self.assertEqual(zero_features.shape, (12, 17, 17))
        self.assertEqual(float(zero_features[OPPONENT_BOMB_ALIGNMENT_CHANNEL].sum()), 0.0)

    def test_control_and_candidate_share_source_and_rewards(self):
        outputs = []
        for arm in ("rule_based_continue_v1", "mixed_kill_v1"):
            environment = os.environ.copy()
            environment["BM_TASK4_TRAINING_ARM"] = arm
            environment["BM_TASK4_TOTAL_EPISODES"] = "2000"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import events as e; "
                        "from agent_code.ben_task4 import callbacks, train; "
                        "print(callbacks.LOAD_MODEL_FILE); "
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                        "print(train.EVENT_REWARDS[e.KILLED_SELF]); "
                        "print(train.TRAINING_OPPONENTS)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs.append(result.stdout.splitlines())

        self.assertEqual(outputs[0][:4], outputs[1][:4])
        self.assertEqual(
            outputs[0][0],
            "ben_task4_baseline_v1_5000ep_seed11.pt",
        )
        self.assertNotEqual(outputs[0][4], outputs[1][4])


if __name__ == "__main__":
    unittest.main()

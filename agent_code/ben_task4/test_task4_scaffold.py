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
    OPPONENT_BOMB_READY_CHANNEL,
    OPPONENT_ESCAPE_PRESSURE_CHANNEL,
    COIN_NAVIGATION_CHANNEL,
    SPACETIME_SAFETY_CHANNEL,
    TEMPORAL_SAFETY_CHANNEL,
    _spacetime_safe_endpoints,
    _coin_navigation_map,
    _temporal_safety_slack,
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
TASK4_INCUMBENT_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task4"
    / "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
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
            "mixed_kill_continue7000_v1": (
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
            "safe_offense_zero_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "safe_offense_reward01_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "safe_offense_reward02_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "opponent_escape_pressure_zero_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "opponent_escape_pressure_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "external_trio_control_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "external_trio_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "opponent_bomb_ready_zero_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "opponent_bomb_ready_v1": (
                "ben_task4_safe_seed11_12ch_alignment_source.pt"
            ),
            "exploration_eps05_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "exploration_eps15_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "exploration_eps05_long3000_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "exploration_eps15_long3000_v1": (
                "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt"
            ),
            "temporal_safety_zero_v1": (
                "ben_task4_mixed_kill_v1_12ch_source.pt"
            ),
            "temporal_safety_v1": (
                "ben_task4_mixed_kill_v1_12ch_source.pt"
            ),
            "spacetime_safety_zero_v1": (
                "ben_task4_mixed_kill_v1_12ch_source.pt"
            ),
            "spacetime_safety_v1": (
                "ben_task4_mixed_kill_v1_12ch_source.pt"
            ),
            "nstep1_control_v1": (
                "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
            ),
            "nstep3_v1": (
                "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
            ),
            "full_symmetry_control_v1": (
                "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
            ),
            "full_symmetry_v1": (
                "ben_task4_mixed_kill_v1_2000ep_seed11.pt"
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

    def test_default_inference_uses_named_incumbent_on_cpu(self):
        self.assertEqual(callbacks.MODEL_VARIANT, "incumbent")
        self.assertEqual(
            callbacks.INFERENCE_MODEL_FILE,
            "ben_task4_mixed_kill_v1_2000ep_seed11.pt",
        )
        agent = SimpleNamespace(train=False, logger=Mock())
        previous_directory = os.getcwd()
        try:
            os.chdir(TASK4_INCUMBENT_MODEL.parent)
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

    def test_baseline_variant_remains_an_explicit_training_source(self):
        environment = os.environ.copy()
        environment["BM_TASK4_MODEL_VARIANT"] = "baseline"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent_code.ben_task4 import callbacks; "
                    "print(callbacks.MODEL_VARIANT); "
                    "print(callbacks.INFERENCE_MODEL_FILE)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["baseline", "ben_task4_baseline_v1_5000ep_seed11.pt"],
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
                "mixed_kill_continue7000_v1",
                "opponent_alignment_mixed_zero_v1",
                "opponent_alignment_mixed_v1",
                "temporal_safety_zero_v1",
                "temporal_safety_v1",
                "nstep1_control_v1",
                "nstep3_v1",
                "full_symmetry_control_v1",
                "full_symmetry_v1",
            }
            else (
                "ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,"
                "ext_xiaoxiae_binary_v6"
                if callbacks.TRAINING_ARM == "external_trio_v1"
                else "rule_based_agent,rule_based_agent,rule_based_agent"
            )
        )
        self.assertEqual(train.TRAINING_OPPONENTS, expected_lineup)

    def test_full_symmetry_pair_differs_only_in_augmentation_mode(self):
        outputs = {}
        for arm in ("full_symmetry_control_v1", "full_symmetry_v1"):
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
                        "print(callbacks.FULL_SYMMETRY_AUGMENTATION); "
                        "print(train.TRAINING_OPPONENTS); "
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.N_STEP_RETURN)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs[arm] = result.stdout.splitlines()

        self.assertEqual(
            outputs["full_symmetry_control_v1"][0],
            "ben_task4_mixed_kill_v1_2000ep_seed11.pt",
        )
        self.assertEqual(
            outputs["full_symmetry_v1"][0],
            "ben_task4_mixed_kill_v1_2000ep_seed11.pt",
        )
        self.assertEqual(outputs["full_symmetry_control_v1"][1], "False")
        self.assertEqual(outputs["full_symmetry_v1"][1], "True")
        self.assertEqual(
            outputs["full_symmetry_control_v1"][2:],
            outputs["full_symmetry_v1"][2:],
        )

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

    def test_safe_offense_reward_requires_bomb_threat_and_escape(self):
        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task4", 0, True, (5, 5)),
            "others": [("opponent", 0, True, (5, 7))],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
        }
        expected = train.SAFE_OFFENSE_BOMB_REWARD
        self.assertEqual(
            train.safe_offense_bomb_reward(game_state, "BOMB"),
            expected,
        )
        self.assertEqual(train.safe_offense_bomb_reward(game_state, "WAIT"), 0.0)

        unavailable_state = dict(game_state)
        unavailable_state["self"] = ("ben_task4", 0, False, (5, 5))
        self.assertEqual(
            train.safe_offense_bomb_reward(unavailable_state, "BOMB"),
            0.0,
        )

        game_state["others"] = [("opponent", 0, True, (9, 9))]
        self.assertEqual(train.safe_offense_bomb_reward(game_state, "BOMB"), 0.0)

        game_state["others"] = [("opponent", 0, True, (5, 6))]
        field[4, 5] = -1
        field[6, 5] = -1
        field[5, 4] = -1
        self.assertEqual(train.safe_offense_bomb_reward(game_state, "BOMB"), 0.0)

        field.fill(0)
        game_state["others"] = [("opponent", 0, True, (5, 8))]
        field[5, 7] = 1
        self.assertEqual(
            train.safe_offense_bomb_reward(game_state, "BOMB"),
            expected,
        )
        field[5, 7] = -1
        self.assertEqual(train.safe_offense_bomb_reward(game_state, "BOMB"), 0.0)

    def test_safe_offense_pair_differs_only_in_auxiliary_reward(self):
        outputs = {}
        for arm in ("safe_offense_zero_v1", "safe_offense_reward02_v1"):
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
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                        "print(train.EVENT_REWARDS[e.KILLED_SELF]); "
                        "print(train.TRAINING_OPPONENTS); "
                        "print(train.SAFE_OFFENSE_BOMB_REWARD)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs[arm] = result.stdout.splitlines()

        zero = outputs["safe_offense_zero_v1"]
        reward = outputs["safe_offense_reward02_v1"]
        self.assertEqual(zero[:-1], reward[:-1])
        self.assertEqual(zero[-1], "0.0")
        self.assertEqual(reward[-1], "0.2")

    def test_reward01_arm_is_the_single_preregistered_intermediate_dose(self):
        outputs = {}
        for arm in ("safe_offense_zero_v1", "safe_offense_reward01_v1"):
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
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                        "print(train.EVENT_REWARDS[e.KILLED_SELF]); "
                        "print(train.TRAINING_OPPONENTS); "
                        "print(train.SAFE_OFFENSE_BOMB_REWARD)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs[arm] = result.stdout.splitlines()

        zero = outputs["safe_offense_zero_v1"]
        reward = outputs["safe_offense_reward01_v1"]
        self.assertEqual(zero[:-1], reward[:-1])
        self.assertEqual(zero[-1], "0.0")
        self.assertEqual(reward[-1], "0.1")

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

    def test_escape_pressure_is_high_for_trapped_aligned_opponent(self):
        field = np.full((17, 17), -1, dtype=np.int8)
        field[5, 5] = 0
        field[5, 6] = 0
        field[5, 7] = 0
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task4", 0, True, (5, 5)),
            "others": [("opponent", 0, True, (5, 7))],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
            "opponent_escape_pressure_mode": "enabled",
        }

        features = state_to_features(game_state)
        pressure = features[OPPONENT_ESCAPE_PRESSURE_CHANNEL]
        self.assertEqual(features.shape, (12, 17, 17))
        self.assertEqual(float(pressure[5, 5]), 1.0)
        self.assertEqual(float(pressure[7, 5]), 0.0)

        field[6, 7] = 0
        field[7, 7] = 0
        field[8, 7] = 0
        escaped_features = state_to_features(game_state)
        self.assertLess(
            float(escaped_features[OPPONENT_ESCAPE_PRESSURE_CHANNEL, 5, 5]),
            1.0,
        )

        game_state["opponent_escape_pressure_mode"] = "zero"
        zero_features = state_to_features(game_state)
        self.assertEqual(
            float(zero_features[OPPONENT_ESCAPE_PRESSURE_CHANNEL].sum()),
            0.0,
        )

    def test_escape_pressure_pair_differs_only_in_feature_mode(self):
        outputs = {}
        for arm in (
            "opponent_escape_pressure_zero_v1",
            "opponent_escape_pressure_v1",
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
                        "print(callbacks.OPPONENT_ESCAPE_PRESSURE_MODE); "
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

        zero = outputs["opponent_escape_pressure_zero_v1"]
        enabled = outputs["opponent_escape_pressure_v1"]
        self.assertEqual(zero[0:3], enabled[0:3])
        self.assertEqual(zero[4:], enabled[4:])
        self.assertEqual(zero[3], "zero")
        self.assertEqual(enabled[3], "enabled")

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

    def test_external_trio_pair_differs_only_in_declared_lineup(self):
        outputs = {}
        for arm in ("external_trio_control_v1", "external_trio_v1"):
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
                        "print(callbacks.ESCAPE_FEATURE_MODE); "
                        "print(callbacks.OPPONENT_ALIGNMENT_MODE); "
                        "print(callbacks.OPPONENT_ESCAPE_PRESSURE_MODE); "
                        "print(train.EVENT_REWARDS[e.COIN_COLLECTED]); "
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

        control = outputs["external_trio_control_v1"]
        candidate = outputs["external_trio_v1"]
        self.assertEqual(control[:-1], candidate[:-1])
        self.assertEqual(
            control[-1],
            "rule_based_agent,rule_based_agent,rule_based_agent",
        )
        self.assertEqual(
            candidate[-1],
            "ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,"
            "ext_xiaoxiae_binary_v6",
        )

    def test_opponent_bomb_ready_channel_marks_only_ready_opponents(self):
        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task4", 0, True, (1, 1)),
            "others": [
                ("ready", 0, True, (5, 7)),
                ("unready", 0, False, (9, 11)),
            ],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
            "opponent_bomb_ready_mode": "enabled",
        }

        features = state_to_features(game_state)
        ready = features[OPPONENT_BOMB_READY_CHANNEL]
        self.assertEqual(features.shape, (12, 17, 17))
        self.assertEqual(float(ready[7, 5]), 1.0)
        self.assertEqual(float(ready[11, 9]), 0.0)
        self.assertEqual(float(ready.sum()), 1.0)

        game_state["opponent_bomb_ready_mode"] = "zero"
        zero_features = state_to_features(game_state)
        self.assertEqual(
            float(zero_features[OPPONENT_BOMB_READY_CHANNEL].sum()),
            0.0,
        )

    def test_opponent_bomb_ready_pair_differs_only_in_feature_mode(self):
        outputs = {}
        for arm in ("opponent_bomb_ready_zero_v1", "opponent_bomb_ready_v1"):
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
                        "print(callbacks.OPPONENT_ESCAPE_PRESSURE_MODE); "
                        "print(callbacks.OPPONENT_BOMB_READY_MODE); "
                        "print(train.EVENT_REWARDS[e.COIN_COLLECTED]); "
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

        zero = outputs["opponent_bomb_ready_zero_v1"]
        enabled = outputs["opponent_bomb_ready_v1"]
        self.assertEqual(zero[0:4], enabled[0:4])
        self.assertEqual(zero[5:], enabled[5:])
        self.assertEqual(zero[4], "zero")
        self.assertEqual(enabled[4], "enabled")

    def test_exploration_pair_differs_only_in_epsilon_start(self):
        outputs = {}
        for arm in ("exploration_eps05_v1", "exploration_eps15_v1"):
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
                        "print(train.LEARNING_RATE); "
                        "print(train.EPSILON_START); "
                        "print(train.EPSILON_END); "
                        "print(train.EPSILON_DECAY_STEPS); "
                        "print(train.EVENT_REWARDS[e.COIN_COLLECTED]); "
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

        control = outputs["exploration_eps05_v1"]
        candidate = outputs["exploration_eps15_v1"]
        self.assertEqual(control[:3], candidate[:3])
        self.assertEqual(control[4:], candidate[4:])
        self.assertEqual(control[3], "0.05")
        self.assertEqual(candidate[3], "0.15")

    def test_long_exploration_pair_starts_fresh_and_differs_only_in_epsilon(self):
        outputs = {}
        for arm in (
            "exploration_eps05_long3000_v1",
            "exploration_eps15_long3000_v1",
        ):
            environment = os.environ.copy()
            environment["BM_TASK4_TRAINING_ARM"] = arm
            environment["BM_TASK4_TOTAL_EPISODES"] = "3000"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from agent_code.ben_task4 import callbacks, train; "
                        "print(callbacks.LOAD_MODEL_FILE); "
                        "print(callbacks.MODEL_FILE); "
                        "print(callbacks.INPUT_CHANNELS); "
                        "print(train.LEARNING_RATE); "
                        "print(train.EPSILON_START); "
                        "print(train.EPSILON_END); "
                        "print(train.EPSILON_DECAY_STEPS); "
                        "print(train.TRAINING_OPPONENTS)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs[arm] = result.stdout.splitlines()

        control = outputs["exploration_eps05_long3000_v1"]
        candidate = outputs["exploration_eps15_long3000_v1"]
        self.assertEqual(control[0], candidate[0])
        self.assertEqual(
            control[0],
            "ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt",
        )
        self.assertNotIn("exploration_eps", control[0])
        self.assertEqual(control[2:4], candidate[2:4])
        self.assertEqual(control[5:], candidate[5:])
        self.assertEqual(control[4], "0.05")
        self.assertEqual(candidate[4], "0.15")
        self.assertIn("3000ep_seed11", control[1])
        self.assertIn("3000ep_seed11", candidate[1])

    def test_temporal_safety_uses_arrival_time_and_bomb_deadline(self):
        field = np.zeros((17, 17), dtype=np.int8)
        explosion_map = np.zeros_like(field)
        slack = _temporal_safety_slack(
            field=field,
            start=(5, 5),
            bombs=[((5, 8), 3)],
            explosion_map=explosion_map,
            opponent_positions=set(),
        )

        self.assertEqual(float(slack[5, 5]), 0.75)
        self.assertEqual(float(slack[5, 6]), 0.5)
        self.assertEqual(float(slack[5, 7]), 0.25)
        self.assertEqual(float(slack[5, 8]), 0.0)
        self.assertEqual(float(slack[6, 6]), 1.0)

        explosion_map[5, 5] = 1
        exploding = _temporal_safety_slack(
            field=field,
            start=(5, 5),
            bombs=[],
            explosion_map=explosion_map,
            opponent_positions=set(),
        )
        self.assertEqual(float(exploding[5, 5]), 0.0)

    def test_temporal_safety_feature_and_zero_control(self):
        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task4", 0, True, (5, 5)),
            "others": [],
            "bombs": [((5, 8), 3)],
            "explosion_map": np.zeros_like(field),
            "temporal_safety_mode": "enabled",
        }
        enabled = state_to_features(game_state)
        self.assertEqual(enabled.shape, (12, 17, 17))
        self.assertGreater(float(enabled[TEMPORAL_SAFETY_CHANNEL].sum()), 0.0)

        game_state["temporal_safety_mode"] = "zero"
        zero = state_to_features(game_state)
        self.assertEqual(zero.shape, (12, 17, 17))
        self.assertEqual(float(zero[TEMPORAL_SAFETY_CHANNEL].sum()), 0.0)

    def test_temporal_safety_pair_differs_only_in_feature_mode(self):
        outputs = {}
        for arm in ("temporal_safety_zero_v1", "temporal_safety_v1"):
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
                        "print(callbacks.TEMPORAL_SAFETY_MODE); "
                        "print(train.EVENT_REWARDS[e.COIN_COLLECTED]); "
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

        zero = outputs["temporal_safety_zero_v1"]
        enabled = outputs["temporal_safety_v1"]
        self.assertEqual(zero[:2], enabled[:2])
        self.assertEqual(zero[3:], enabled[3:])
        self.assertEqual(zero[2], "zero")
        self.assertEqual(enabled[2], "enabled")
        self.assertEqual(
            zero[-1],
            "peaceful_agent,rule_based_agent,rule_based_agent",
        )

    def test_temporal_safety_replay_state_shapes_match(self):
        environment = os.environ.copy()
        environment["BM_TASK4_TRAINING_ARM"] = "temporal_safety_zero_v1"
        environment["BM_TASK4_TOTAL_EPISODES"] = "1000"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from types import SimpleNamespace; import numpy as np; "
                    "from agent_code.ben_task4 import train; "
                    "f=np.zeros((17,17),dtype=np.int8); "
                    "s={'round':1,'step':2,'field':f,'coins':[],"
                    "'self':('ben_task4',0,True,(5,5)),'others':[],"
                    "'bombs':[((5,8),3)],'explosion_map':np.zeros_like(f)}; "
                    "a=SimpleNamespace(last_action_features=None,"
                    "last_feature_round_step=None,"
                    "visit_counts=np.zeros_like(f,dtype=np.float32)); "
                    "print(train._features_for_stored_action(a,s).shape); "
                    "print(train._next_features_with_visit_count(a,s).shape)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.stdout.splitlines(), ["(12, 17, 17)"] * 2)

    def test_spacetime_safety_waits_for_active_explosion_to_fade(self):
        field = np.full((17, 17), -1, dtype=np.int8)
        field[1:5, 5] = 0
        explosion_map = np.zeros_like(field)
        explosion_map[2, 5] = 2

        endpoints = _spacetime_safe_endpoints(
            field=field,
            start=(1, 5),
            bombs=[],
            explosion_map=explosion_map,
            opponent_positions=set(),
        )

        # Waiting at (1, 5) for one tick and then crossing is safe. A static
        # BFS through the active explosion would use it too early, while a
        # no-wait search would miss this valid route.
        self.assertAlmostEqual(float(endpoints[3, 5]), 0.5)

    def test_spacetime_safety_feature_and_zero_control(self):
        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task4", 0, True, (5, 5)),
            "others": [],
            "bombs": [((5, 8), 3)],
            "explosion_map": np.zeros_like(field),
            "spacetime_safety_mode": "enabled",
        }
        enabled = state_to_features(game_state)
        self.assertEqual(enabled.shape, (12, 17, 17))
        self.assertGreater(
            float(enabled[SPACETIME_SAFETY_CHANNEL].sum()),
            0.0,
        )

        game_state["spacetime_safety_mode"] = "zero"
        zero = state_to_features(game_state)
        self.assertEqual(zero.shape, (12, 17, 17))
        self.assertEqual(float(zero[SPACETIME_SAFETY_CHANNEL].sum()), 0.0)

    def test_spacetime_safety_action_features_match_12_channel_model(self):
        environment = os.environ.copy()
        environment["BM_TASK4_TRAINING_ARM"] = "spacetime_safety_zero_v1"
        environment["BM_TASK4_TOTAL_EPISODES"] = "1000"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from types import SimpleNamespace; import numpy as np; "
                    "from agent_code.ben_task4 import callbacks; "
                    "f=np.zeros((17,17),dtype=np.int8); "
                    "s={'round':1,'step':1,'field':f,'coins':[],"
                    "'self':('ben_task4',0,True,(5,5)),'others':[],"
                    "'bombs':[],'explosion_map':np.zeros_like(f)}; "
                    "a=SimpleNamespace(visit_round=None,visit_counts=None,"
                    "visit_count_encoding=callbacks.VISIT_COUNT_ENCODING,"
                    "last_action_features=None,last_feature_round_step=None); "
                    "print(callbacks._features_with_visit_count(a,s).shape)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.stdout.strip(), "(12, 17, 17)")

    def test_coin_navigation_uses_walkable_paths_and_blocks_bombs(self):
        field = np.zeros((17, 17), dtype=np.int8)
        field[2, 1] = -1
        navigation = _coin_navigation_map(
            field, [(3, 1)], [((1, 2), 3)], set()
        )
        self.assertEqual(float(navigation[1, 2]), 0.0)
        self.assertGreater(float(navigation[1, 1]), 0.0)
        self.assertLess(float(navigation[1, 1]), 1.0)

    def test_coin_navigation_feature_and_zero_control(self):
        field = np.zeros((17, 17), dtype=np.int8)
        state = {
            "field": field, "coins": [(8, 8)],
            "self": ("ben_task4", 0, True, (5, 5)), "others": [],
            "bombs": [], "explosion_map": np.zeros_like(field),
            "coin_navigation_mode": "enabled",
        }
        enabled = state_to_features(state)
        self.assertGreater(float(enabled[COIN_NAVIGATION_CHANNEL].sum()), 0.0)
        state["coin_navigation_mode"] = "zero"
        self.assertEqual(float(state_to_features(state)[COIN_NAVIGATION_CHANNEL].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

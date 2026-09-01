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
import torch
import events as e

from agent_code.ben_task3 import callbacks, train
from agent_code.ben_task3.features import (
    OPPONENT_CHANNEL,
    legal_action_mask,
    state_to_features,
)
from agent_code.ben_task3.model import CoinCollectorDQN


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK2_FROZEN_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "dqn_task2_coin_reward15"
    / "dqn_task2_coin_reward15_seed11.pt"
)
TASK3_BASELINE_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task3"
    / "ben_task3_task2_baseline_11ch_seed11.pt"
)
TASK3_SOURCE_MODEL = (
    REPO_ROOT
    / "agent_code"
    / "ben_task3"
    / "ben_task3_task2_baseline_seed11.pt"
)


class Task3ScaffoldTest(unittest.TestCase):
    def test_local_source_is_byte_identical_to_frozen_task2(self):
        self.assertEqual(
            hashlib.sha256(TASK2_FROZEN_MODEL.read_bytes()).hexdigest(),
            hashlib.sha256(TASK3_SOURCE_MODEL.read_bytes()).hexdigest(),
        )

    def test_converted_baseline_preserves_old_channels_and_zeros_opponent(self):
        source = torch.load(
            TASK3_SOURCE_MODEL,
            map_location="cpu",
            weights_only=True,
        )
        converted = torch.load(
            TASK3_BASELINE_MODEL,
            map_location="cpu",
            weights_only=True,
        )
        first_weight = "convolutional.0.weight"

        self.assertEqual(converted[first_weight].shape[1], 11)
        self.assertTrue(
            torch.equal(
                converted[first_weight][:, :10],
                source[first_weight],
            )
        )
        self.assertEqual(
            float(converted[first_weight][:, 10].abs().sum()),
            0.0,
        )
        for key in source:
            if key != first_weight:
                self.assertTrue(torch.equal(converted[key], source[key]))

    def test_artifact_names_and_paths_are_task3_only(self):
        self.assertEqual(
            callbacks.MODEL_FILE,
            "ben_task3_peaceful_v1_5000ep_seed11.pt",
        )
        self.assertEqual(
            callbacks.LOAD_MODEL_FILE,
            "ben_task3_task2_baseline_11ch_seed11.pt",
        )
        self.assertEqual(
            train.RUN_LABEL,
            "task3_peaceful_v1_5000ep_seed11",
        )
        self.assertEqual(train.TRAINLOG_OUT_DIR, "results/train/ben_task3")
        self.assertEqual(train.CHECKPOINT_DIR, "../../results/train/ben_task3")
        self.assertNotIn("ben_task2", callbacks.MODEL_FILE)
        self.assertNotIn("ben_task2", train.RUN_LABEL)
        self.assertNotIn("ben_task2", train.TRAINLOG_OUT_DIR)

    def test_inference_falls_back_to_local_task2_baseline_on_cpu(self):
        agent = SimpleNamespace(train=False, logger=Mock())
        previous_directory = os.getcwd()
        try:
            os.chdir(TASK3_BASELINE_MODEL.parent)
            callbacks.setup(agent)
        finally:
            os.chdir(previous_directory)

        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "round": 1,
            "step": 1,
            "field": field,
            "coins": [(5, 5)],
            "self": ("ben_task3", 0, True, (1, 1)),
            "bombs": [],
            "others": [("peaceful", 0, False, (5, 1))],
            "explosion_map": np.zeros_like(field),
        }

        self.assertEqual(agent.device.type, "cpu")
        self.assertIn(callbacks.act(agent, game_state), callbacks.ACTIONS)
        self.assertEqual(agent.last_action_features.shape, (11, 17, 17))

    def test_training_output_does_not_reuse_baseline_name(self):
        self.assertNotEqual(callbacks.MODEL_FILE, callbacks.LOAD_MODEL_FILE)
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertFalse(
                (Path(temporary_directory) / callbacks.MODEL_FILE).exists()
            )

    def test_opponent_channel_marks_raw_positions(self):
        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task3", 0, True, (1, 1)),
            "others": [
                ("peaceful-1", 0, False, (5, 1)),
                ("peaceful-2", 0, False, (7, 3)),
            ],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
            "escape_feature_mode": "reachable_safe_tiles",
        }

        features = state_to_features(game_state)

        self.assertEqual(features.shape, (11, 17, 17))
        self.assertEqual(float(features[OPPONENT_CHANNEL].sum()), 2.0)
        self.assertEqual(float(features[OPPONENT_CHANNEL, 1, 5]), 1.0)
        self.assertEqual(float(features[OPPONENT_CHANNEL, 3, 7]), 1.0)

    def test_opponent_tile_is_not_a_legal_movement_target(self):
        field = np.zeros((17, 17), dtype=np.int8)
        game_state = {
            "field": field,
            "coins": [],
            "self": ("ben_task3", 0, True, (1, 1)),
            "others": [("peaceful", 0, False, (2, 1))],
            "bombs": [],
            "explosion_map": np.zeros_like(field),
            "escape_feature_mode": "reachable_safe_tiles",
        }

        mask = legal_action_mask(state_to_features(game_state))

        self.assertFalse(bool(mask[1]))

    def test_task3_kill_reward_matches_official_score(self):
        self.assertEqual(train.EVENT_REWARDS[e.KILLED_OPPONENT], 5.0)
        self.assertAlmostEqual(
            train.reward_from_events([e.KILLED_OPPONENT]),
            4.95,
        )

    def test_official_training_score_includes_five_points_per_kill(self):
        events = [e.COIN_COLLECTED] * 2 + [e.KILLED_OPPONENT] * 3
        self.assertEqual(train._official_score(events), 17)

    def test_task3_model_accepts_eleven_channels(self):
        model = CoinCollectorDQN(input_channels=11)
        output = model(torch.zeros((2, 11, 17, 17)))
        self.assertEqual(output.shape, (2, 6))

    def test_model_variant_selects_local_baseline_without_changing_output(self):
        environment = os.environ.copy()
        environment["BM_TASK3_MODEL_VARIANT"] = "baseline"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent_code.ben_task3 import callbacks; "
                    "print(callbacks.INFERENCE_MODEL_FILE); "
                    "print(callbacks.MODEL_FILE)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ben_task3_task2_baseline_11ch_seed11.pt",
                "ben_task3_peaceful_v1_5000ep_seed11.pt",
            ],
        )

    def test_coin_collector_arm_uses_new_paths_and_peaceful_source(self):
        environment = os.environ.copy()
        environment["BM_TASK3_TRAINING_ARM"] = (
            "coin_collector_finetune_v1"
        )
        environment["BM_TASK3_TOTAL_EPISODES"] = "2000"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent_code.ben_task3 import callbacks, train; "
                    "print(callbacks.LOAD_MODEL_FILE); "
                    "print(callbacks.MODEL_FILE); "
                    "print(train.RUN_LABEL)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ben_task3_peaceful_v1_5000ep_seed11.pt",
                "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt",
                "task3_coin_collector_finetune_v1_2000ep_seed11",
            ],
        )

    def test_kill_reward_arm_changes_only_named_reward_and_paths(self):
        environment = os.environ.copy()
        environment["BM_TASK3_TRAINING_ARM"] = (
            "coin_collector_kill_reward75_v1"
        )
        environment["BM_TASK3_TOTAL_EPISODES"] = "2000"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import events as e; "
                    "from agent_code.ben_task3 import callbacks, train; "
                    "print(callbacks.LOAD_MODEL_FILE); "
                    "print(callbacks.MODEL_FILE); "
                    "print(train.RUN_LABEL); "
                    "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                    "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                    "print(train.EVENT_REWARDS[e.COIN_COLLECTED])"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt",
                "ben_task3_coin_collector_kill_reward75_v1_2000ep_seed11.pt",
                "task3_coin_collector_kill_reward75_v1_2000ep_seed11",
                "7.5",
                "-5.0",
                "1.5",
            ],
        )

    def test_opponent_potential_arm_uses_safe_source_and_dense_shaping(self):
        environment = os.environ.copy()
        environment["BM_TASK3_TRAINING_ARM"] = (
            "coin_collector_opponent_potential_v1"
        )
        environment["BM_TASK3_TOTAL_EPISODES"] = "2000"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import numpy as np; "
                    "from agent_code.ben_task3 import callbacks, train; "
                    "f=np.zeros((17,17),dtype=np.int8); "
                    "s=lambda x:{'field':f,'self':('b',0,True,(x,1)),"
                    "'others':[('o',0,True,(5,1))],'step':1}; "
                    "print(callbacks.LOAD_MODEL_FILE); "
                    "print(callbacks.MODEL_FILE); "
                    "print(train.EVENT_REWARDS['KILLED_OPPONENT']); "
                    "print(train.OPPONENT_POTENTIAL_SCALE); "
                    "print(train._shortest_opponent_distance(s(1))); "
                    "print(train.opponent_potential_shaping_reward(s(1),s(2))>0); "
                    "print(train.opponent_potential_shaping_reward(s(2),s(1))<0)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt",
                "ben_task3_coin_collector_opponent_potential_v1_2000ep_seed11.pt",
                "5.0",
                "1.0",
                "4",
                "True",
                "True",
            ],
        )

    def test_checkpoint_selector_uses_existing_relative_task3_path(self):
        environment = os.environ.copy()
        environment["BM_TASK3_TRAINING_ARM"] = (
            "coin_collector_opponent_potential_v1"
        )
        environment["BM_TASK3_TOTAL_EPISODES"] = "2000"
        environment["BM_TASK3_MODEL_VARIANT"] = "trained"
        environment["BM_TASK3_CHECKPOINT_EPISODE"] = "1200"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agent_code.ben_task3 import callbacks; "
                    "print(callbacks.INFERENCE_MODEL_FILE); "
                    "print(callbacks.MODEL_FILE); "
                    "print(callbacks.CHECKPOINT_EPISODE)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "../../results/train/ben_task3/task3_coin_collector_opponent_potential_v1_2000ep_seed11__episode_1200.pt",
                "ben_task3_coin_collector_opponent_potential_v1_2000ep_seed11.pt",
                "1200",
            ],
        )

    def test_mixed_curriculum_uses_safe_source_and_unchanged_rewards(self):
        environment = os.environ.copy()
        environment["BM_TASK3_TRAINING_ARM"] = "mixed_curriculum_v1"
        environment["BM_TASK3_TOTAL_EPISODES"] = "2000"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import events as e; "
                    "from agent_code.ben_task3 import callbacks, train; "
                    "print(callbacks.LOAD_MODEL_FILE); "
                    "print(callbacks.MODEL_FILE); "
                    "print(train.RUN_LABEL); "
                    "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                    "print(train.EVENT_REWARDS[e.GOT_KILLED]); "
                    "print(train.OPPONENT_POTENTIAL_SCALE); "
                    "print(train.TRAINING_OPPONENTS)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt",
                "ben_task3_mixed_curriculum_v1_2000ep_seed11.pt",
                "task3_mixed_curriculum_v1_2000ep_seed11",
                "5.0",
                "-5.0",
                "0.0",
                "peaceful_agent,coin_collector_agent,coin_collector_agent",
            ],
        )

    def test_multiseed_safety_arm_uses_mixed_source_and_active_opponents(self):
        for training_seed in (11, 12, 13):
            environment = os.environ.copy()
            environment["BM_TASK3_TRAINING_ARM"] = (
                "mixed_safety_multiseed_v1"
            )
            environment["BM_TASK3_TRAINING_SEED"] = str(training_seed)
            environment["BM_TASK3_TOTAL_EPISODES"] = "5000"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import events as e; "
                        "from agent_code.ben_task3 import callbacks, train; "
                        "print(callbacks.LOAD_MODEL_FILE); "
                        "print(callbacks.MODEL_FILE); "
                        "print(train.RUN_LABEL); "
                        "print(train.EVENT_REWARDS[e.KILLED_OPPONENT]); "
                        "print(train.OPPONENT_POTENTIAL_SCALE); "
                        "print(train.TRAINING_OPPONENTS)"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                [
                    "ben_task3_mixed_curriculum_v1_2000ep_seed11.pt",
                    f"ben_task3_mixed_safety_multiseed_v1_5000ep_seed{training_seed}.pt",
                    f"task3_mixed_safety_multiseed_v1_5000ep_seed{training_seed}",
                    "5.0",
                    "0.0",
                    "coin_collector_agent,coin_collector_agent,coin_collector_agent",
                ],
            )


if __name__ == "__main__":
    unittest.main()

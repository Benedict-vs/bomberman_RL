"""Unit tests for Task-4 training-state checkpoint/resume semantics."""

from __future__ import annotations

from collections import deque
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
import torch

from agent_code.ben_task4 import train
from agent_code.ben_task4.dqn import update_target_network
from agent_code.ben_task4.model import CoinCollectorDQN
from agent_code.ben_task4.replay_buffer import ReplayBuffer


def _agent() -> SimpleNamespace:
    online = CoinCollectorDQN(input_channels=train.INPUT_CHANNELS)
    target = CoinCollectorDQN(input_channels=train.INPUT_CHANNELS)
    update_target_network(online, target)
    optimizer = torch.optim.Adam(online.parameters(), lr=train.LEARNING_RATE)
    loss = online(torch.zeros((1, train.INPUT_CHANNELS, 17, 17))).sum()
    loss.backward()
    optimizer.step()

    replay = ReplayBuffer(
        capacity=train.REPLAY_CAPACITY,
        event_balance=train.EVENT_REPLAY_MODE == "balanced",
        prioritized=train.PRIORITIZED_REPLAY,
    )
    features = np.zeros((train.INPUT_CHANNELS, 17, 17), dtype=np.float32)
    replay.append(features, 1, 0.5, features, False)
    return SimpleNamespace(
        online_network=online,
        target_network=target,
        optimizer=optimizer,
        replay_buffer=replay,
        n_step_queue=deque(),
        environment_steps=5_432,
        optimization_steps=123,
        epsilon=0.071,
        episode_offset=0,
        device=torch.device("cpu"),
        logger=Mock(),
    )


class TrainingResumeTest(unittest.TestCase):
    def test_training_state_round_trip_restores_learning_state(self) -> None:
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        source = _agent()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        expected_python = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = float(torch.rand(()))
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        expected_online = {
            name: value.detach().clone()
            for name, value in source.online_network.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "resume_state.pt"
            original_output = train.TRAINING_STATE_FILE
            try:
                train.TRAINING_STATE_FILE = str(state_path)
                train._save_training_state(source, episode=700)
            finally:
                train.TRAINING_STATE_FILE = original_output

            restored = _agent()
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            with patch.object(train.torch, "load", wraps=train.torch.load) as load:
                train._restore_training_state(restored, str(state_path))
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")

        self.assertEqual(restored.episode_offset, 700)
        self.assertEqual(restored.environment_steps, 5_432)
        self.assertEqual(restored.optimization_steps, 123)
        self.assertEqual(restored.epsilon, 0.071)
        self.assertEqual(len(restored.replay_buffer), 1)
        self.assertTrue(restored.optimizer.state_dict()["state"])
        for name, value in restored.online_network.state_dict().items():
            self.assertTrue(torch.equal(value, expected_online[name]))
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)
        self.assertEqual(float(torch.rand(())), expected_torch)

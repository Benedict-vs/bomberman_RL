"""Equivalence tests for the frozen inference-only Task-2 baseline."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from agent_code.dqn_task2 import callbacks as frozen
from agent_code.ben_task2_visit_count_eval import callbacks as reference


class FrozenTask2BaselineTest(unittest.TestCase):
    def test_frozen_model_is_byte_identical(self) -> None:
        source = Path(
            "agent_code/ben_task2/ben_task2_visit_count_v1_7000ep_seed11.pt"
        )
        target = Path(
            "agent_code/dqn_task2/dqn_task2_visit_count_v1_7000ep_seed11.pt"
        )
        self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_frozen_and_reference_agents_choose_same_actions(self) -> None:
        reference_agent = SimpleNamespace(train=False, logger=Mock())
        frozen_agent = SimpleNamespace(train=False, logger=Mock())
        previous_directory = os.getcwd()
        try:
            os.chdir("agent_code/ben_task2_visit_count_eval")
            reference.setup(reference_agent)
            os.chdir("../dqn_task2")
            frozen.setup(frozen_agent)
        finally:
            os.chdir(previous_directory)

        field = np.zeros((17, 17), dtype=np.int8)
        field[3, 1] = 1
        state = {
            "round": 1,
            "step": 1,
            "field": field,
            "coins": [(5, 1)],
            "self": ("agent", 0, True, (1, 1)),
            "bombs": [((1, 3), 2)],
            "explosion_map": np.zeros_like(field),
            "others": [],
        }

        for step in range(1, 6):
            state["step"] = step
            self.assertEqual(
                reference.act(reference_agent, state),
                frozen.act(frozen_agent, state),
            )


if __name__ == "__main__":
    unittest.main()

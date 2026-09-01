"""Tests for the frozen visit-count evaluation wrapper."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from agent_code.ben_task2_visit_count_eval import callbacks


class VisitCountEvaluationCallbacksTest(unittest.TestCase):
    def test_visit_channel_counts_and_resets_between_rounds(self) -> None:
        agent = SimpleNamespace(visit_round=None, visit_counts=None)
        field = np.zeros((17, 17), dtype=int)
        state = {
            "round": 1,
            "step": 1,
            "field": field,
            "self": ("ben", 0, True, (3, 4)),
            "coins": [],
            "bombs": [],
            "others": [],
            "explosion_map": np.zeros_like(field),
        }

        first = callbacks._features_with_visit_count(agent, state)
        second = callbacks._features_with_visit_count(agent, state)
        state["round"] = 2
        reset = callbacks._features_with_visit_count(agent, state)

        self.assertAlmostEqual(float(first[8, 4, 3]), 0.1)
        self.assertAlmostEqual(float(second[8, 4, 3]), 0.2)
        self.assertAlmostEqual(float(reset[8, 4, 3]), 0.1)

    def test_frozen_model_matches_training_artifact(self) -> None:
        source = torch.load(
            "agent_code/ben_task2/ben_task2_visit_count_v1_7000ep_seed11.pt",
            map_location="cpu",
            weights_only=True,
        )
        frozen = torch.load(
            "agent_code/ben_task2_visit_count_eval/"
            "ben_task2_visit_count_v1_7000ep_seed11.pt",
            map_location="cpu",
            weights_only=True,
        )

        self.assertEqual(source.keys(), frozen.keys())
        for key in source:
            self.assertTrue(torch.equal(source[key], frozen[key]), key)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Q-preserving own-bomb outcome auxiliary head."""

import unittest

import numpy as np
import torch

from agent_code.ben_task4 import dqn
from agent_code.ben_task4.model import (
    BombOutcomeDQN,
    CoinCollectorDQN,
    bomb_outcome_state_dict_from_classic,
)
from agent_code.ben_task4.replay_buffer import BombOutcomeExample, Transition


class BombOutcomeModelTest(unittest.TestCase):
    def test_conversion_preserves_q_values_and_exposes_three_logits(self):
        torch.manual_seed(7)
        classic = CoinCollectorDQN()
        auxiliary = BombOutcomeDQN()
        auxiliary.load_state_dict(
            bomb_outcome_state_dict_from_classic(classic.state_dict())
        )
        states = torch.randn(3, 11, 17, 17)
        torch.testing.assert_close(classic(states), auxiliary(states), rtol=0, atol=1e-6)
        _q_values, logits = auxiliary.forward_with_bomb_outcome(states)
        self.assertEqual(tuple(logits.shape), (3, 3))

    def test_empty_auxiliary_batch_leaves_the_dqn_update_unchanged(self):
        torch.manual_seed(13)
        control = BombOutcomeDQN()
        candidate = BombOutcomeDQN()
        candidate.load_state_dict(control.state_dict())
        target = BombOutcomeDQN()
        target.load_state_dict(control.state_dict())
        transition = _terminal_transition()

        control_loss = dqn.optimize_dqn(
            control, target, torch.optim.SGD(control.parameters(), lr=0.01),
            [transition], 0.99, torch.device("cpu"),
        )
        candidate_loss = dqn.optimize_dqn(
            candidate, target, torch.optim.SGD(candidate.parameters(), lr=0.01),
            [transition], 0.99, torch.device("cpu"),
            auxiliary_bomb_outcome=True, bomb_outcome_examples=[],
        )

        self.assertEqual(control_loss, candidate_loss)
        for control_parameter, candidate_parameter in zip(
            control.parameters(), candidate.parameters()
        ):
            torch.testing.assert_close(control_parameter, candidate_parameter)

    def test_resolved_outcome_batch_updates_the_auxiliary_head(self):
        torch.manual_seed(19)
        online = BombOutcomeDQN()
        target = BombOutcomeDQN()
        target.load_state_dict(online.state_dict())
        before = online.bomb_outcome_head.bias.detach().clone()
        outcome_example = BombOutcomeExample(
            state=np.zeros((11, 17, 17), dtype=np.float32), outcome=1,
        )

        loss = dqn.optimize_dqn(
            online, target, torch.optim.SGD(online.parameters(), lr=0.01),
            [_terminal_transition()], 0.99, torch.device("cpu"),
            auxiliary_bomb_outcome=True,
            bomb_outcome_examples=[outcome_example],
        )

        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, online.bomb_outcome_head.bias))


def _terminal_transition() -> Transition:
    state = np.zeros((11, 17, 17), dtype=np.float32)
    return Transition(
        state=state,
        action=0,
        reward=1.0,
        next_state=np.zeros_like(state),
        done=True,
    )


if __name__ == "__main__":
    unittest.main()

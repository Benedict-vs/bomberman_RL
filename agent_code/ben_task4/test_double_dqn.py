import unittest

import numpy as np
import torch
from torch import nn

from agent_code.ben_task4 import dqn
from agent_code.ben_task4.replay_buffer import Transition


class FixedQNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.unsqueeze(0).expand(len(states), -1)


class DoubleDQNTest(unittest.TestCase):
    def test_double_dqn_selects_online_action_but_uses_target_value(self):
        transition = Transition(
            state=np.zeros((1, 1, 1), dtype=np.float32),
            action=0,
            reward=0.0,
            next_state=np.zeros((1, 1, 1), dtype=np.float32),
            done=False,
        )
        old_mask = dqn.legal_action_mask
        dqn.legal_action_mask = lambda _state: np.array([True, True])
        try:
            online = FixedQNetwork([0.0, 5.0])
            target = FixedQNetwork([10.0, 1.0])
            optimizer = torch.optim.SGD(online.parameters(), lr=0.0)
            classic_loss = dqn.optimize_dqn(
                online, target, optimizer, [transition], 1.0, torch.device("cpu")
            )
            double_loss = dqn.optimize_dqn(
                online, target, optimizer, [transition], 1.0, torch.device("cpu"),
                double_dqn=True,
            )
        finally:
            dqn.legal_action_mask = old_mask

        self.assertAlmostEqual(classic_loss, 9.5)
        self.assertAlmostEqual(double_loss, 0.5)


if __name__ == "__main__":
    unittest.main()

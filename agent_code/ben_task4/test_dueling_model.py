import unittest

import torch

from agent_code.ben_task4.model import (
    CoinCollectorDQN,
    DuelingCoinCollectorDQN,
    dueling_state_dict_from_classic,
)


class DuelingModelTest(unittest.TestCase):
    def test_conversion_preserves_all_q_values(self):
        torch.manual_seed(11)
        classic = CoinCollectorDQN()
        dueling = DuelingCoinCollectorDQN()
        dueling.load_state_dict(dueling_state_dict_from_classic(classic.state_dict()))
        states = torch.randn(3, 11, 17, 17)
        torch.testing.assert_close(classic(states), dueling(states), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

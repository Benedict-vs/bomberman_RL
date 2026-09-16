import unittest
import torch

from agent_code.ben_task4.model import (
    CoinCollectorDQN,
    OpponentPredictionDQN,
    opponent_prediction_state_dict_from_classic,
)


class OpponentPredictionModelTest(unittest.TestCase):
    def test_conversion_preserves_q_values(self):
        torch.manual_seed(11)
        classic = CoinCollectorDQN()
        auxiliary = OpponentPredictionDQN()
        auxiliary.load_state_dict(opponent_prediction_state_dict_from_classic(classic.state_dict()))
        states = torch.randn(2, 11, 17, 17)
        torch.testing.assert_close(classic(states), auxiliary(states), rtol=0, atol=1e-6)
        _q, logits = auxiliary.forward_with_opponent_prediction(states)
        self.assertEqual(tuple(logits.shape), (2, 17, 17))


if __name__ == "__main__":
    unittest.main()

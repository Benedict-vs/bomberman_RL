import unittest

import torch

from agent_code.ben_task2.model import CoinCollectorDQN
from tools.convert_safety_model_9ch import (
    FIRST_CONVOLUTION_WEIGHT,
    convert_state_dict,
)


class SafetyCompatibilityTest(unittest.TestCase):
    def test_conversion_preserves_old_weights_and_zeros_new_channel(self):
        target_state = CoinCollectorDQN().state_dict()
        old_state = {
            name: tensor.clone()
            for name, tensor in target_state.items()
        }
        old_state[FIRST_CONVOLUTION_WEIGHT] = old_state[
            FIRST_CONVOLUTION_WEIGHT
        ][:, :8].clone()

        converted = convert_state_dict(old_state)

        torch.testing.assert_close(
            converted[FIRST_CONVOLUTION_WEIGHT][:, :8],
            old_state[FIRST_CONVOLUTION_WEIGHT],
        )
        self.assertTrue(
            torch.count_nonzero(
                converted[FIRST_CONVOLUTION_WEIGHT][:, 8]
            ).item() == 0
        )

        for name in old_state:
            if name != FIRST_CONVOLUTION_WEIGHT:
                torch.testing.assert_close(converted[name], old_state[name])


if __name__ == "__main__":
    unittest.main()

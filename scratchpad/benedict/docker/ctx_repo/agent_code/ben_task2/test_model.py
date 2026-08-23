import unittest

import torch

from model import ACTIONS, N_ACTIONS, N_INPUT_CHANNELS, CoinCollectorDQN


class CoinCollectorDQNTest(unittest.TestCase):
    def test_forward_pass_shape(self):
        model = CoinCollectorDQN()

        batch = torch.zeros(
            (4, N_INPUT_CHANNELS, 17, 17),
            dtype=torch.float32,
        )

        q_values = model(batch)

        self.assertEqual(q_values.shape, (4, N_ACTIONS))
        self.assertEqual(q_values.dtype, torch.float32)
        self.assertTrue(torch.isfinite(q_values).all())

    def test_action_space(self):
        self.assertEqual(
            ACTIONS,
            ("UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"),
        )
        self.assertEqual(N_ACTIONS, 6)

    def test_parameter_count(self):
        model = CoinCollectorDQN()

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        self.assertEqual(parameter_count, 118_358)

    def test_backward_pass(self):
        model = CoinCollectorDQN()

        batch = torch.rand(
            (2, N_INPUT_CHANNELS, 17, 17),
            dtype=torch.float32,
        )

        q_values = model(batch)
        loss = q_values.square().mean()
        loss.backward()

        parameters_with_gradients = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

        self.assertGreater(len(parameters_with_gradients), 0)


if __name__ == "__main__":
    unittest.main()

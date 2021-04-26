import torch
from torch.optim import Adam
import source.networks.critic as C
import unittest


class CriticTest(unittest.TestCase):

    def setUp(self) -> None:
        self.seed = 42
        self.num_state = 5
        self.num_action = 2
        self.batch_size = 10
        self.n_estimates = 2

        self.critic = C.Critic(self.num_state, self.num_action, self.seed, self.n_estimates)

    def test_shapes(self):
        input_1 = torch.randn((self.batch_size, self.num_state))
        input_2 = torch.randn((self.num_state))

        out_1 = self.critic(input_1)

        assert out_1.shape == (self.batch_size, self.num_action * self.n_estimates)

        out_2 = self.critic(input_2)

        assert out_2.shape == (1, self.num_action * self.n_estimates)

    def test_correct_seed(self):
        new_critic = C.Critic(self.num_state, self.num_action, self.seed, self.n_estimates)

        # test equality of first forward pass
        input = torch.randn((self.batch_size, self.num_state))

        output_1 = self.critic(input)
        output_2 = new_critic(input)

        assert torch.allclose(output_1, output_2)

        # test equality of network after first backprop

        optimizer_1 = Adam(self.critic.parameters(), lr=1e-3)
        optimizer_2 = Adam(new_critic.parameters(), lr=1e-3)

        output_1.mean().backward()
        optimizer_1.step()

        output_2.mean().backward()
        optimizer_2.step()

        output_1 = self.critic(input)
        output_2 = new_critic(input)

        assert torch.allclose(output_1, output_2)






# rl/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import NOISY_SIGMA  


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mu_range = 1 / np.sqrt(in_features)

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-mu_range, mu_range)
        )
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma_init / np.sqrt(in_features))
        )

        self.bias_mu = nn.Parameter(
            torch.empty(out_features).uniform_(-mu_range, mu_range)
        )
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma_init / np.sqrt(in_features))
        )

    def forward(self, x):
        if self.training:
            eps_in = torch.randn(self.in_features, device=x.device)
            eps_out = torch.randn(self.out_features, device=x.device)

            f_in = torch.sign(eps_in) * torch.sqrt(torch.abs(eps_in))
            f_out = torch.sign(eps_out) * torch.sqrt(torch.abs(eps_out))

            w_noise = torch.ger(f_out, f_in)
            b_noise = f_out

            weight = self.weight_mu + self.weight_sigma * w_noise
            bias = self.bias_mu + self.bias_sigma * b_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return x @ weight.t() + bias


class RainbowDQN(nn.Module):
    def __init__(self, state_size, num_actions, num_atoms, v_min, v_max):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # Shared feature extractor
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

        # Value stream
        self.v_noisy1 = NoisyLinear(128, 128)
        self.v_noisy2 = NoisyLinear(128, num_atoms)

        # Advantage stream
        self.a_noisy1 = NoisyLinear(128, 128)
        self.a_noisy2 = NoisyLinear(128, num_actions * num_atoms)

        self.relu = nn.ReLU()

        # C51 support
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms)
        )

    def forward(self, x):
        # Shared layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Value stream
        v = self.relu(self.v_noisy1(x))
        v = self.v_noisy2(v)

        # Advantage stream
        a = self.relu(self.a_noisy1(x))
        a = self.a_noisy2(a)
        a = a.view(-1, self.num_actions, self.num_atoms)

        # Combine streams (dueling)
        v = v.view(-1, 1, self.num_atoms)
        a_mean = a.mean(dim=1, keepdim=True)

        return v + (a - a_mean)

    def q_values(self, x):
        probs = torch.softmax(self.forward(x), dim=-1)
        return torch.sum(probs * self.support, dim=-1)


def build_rainbow_model(state_size, num_actions, num_atoms, v_min, v_max, lr, device):
    model = RainbowDQN(state_size, num_actions, num_atoms, v_min, v_max).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    return model, optimizer

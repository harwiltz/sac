import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self._l1 = nn.Linear(input_dim, 256)
        self._l2 = nn.Linear(256, 256)
        self._l3 = nn.Linear(256, 1)
        self._input_dim = input_dim

    def forward(self, x):
        x = F.relu(self._l1(x))
        x = F.relu(self._l2(x))
        x = self._l3(x)
        return x

    def exponential_smooth(self, other, tau):
        for (self_p, other_p) in zip(self.parameters(), other.parameters()):
            self_p.data.copy_(self_p.data * (1. - tau) + other_p.data * tau)

class CriticNetwork(ValueNetwork):
    def __init__(self, obs_dim, act_dim):
        super(CriticNetwork, self).__init__(obs_dim + act_dim)
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return ValueNetwork.forward(self, x)

class ActorNetwork(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_low,
                 act_high,
                 log_std_min=-20,
                 log_std_max=20):
        super(ActorNetwork, self).__init__()
        self._l1 = nn.Linear(obs_dim, 256)
        self._l2 = nn.Linear(256, 256)
        self._mean_layer = nn.Linear(256, act_dim)
        self._std_layer = nn.Linear(256, act_dim)
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        act_scale = torch.FloatTensor(act_high - act_low)
        act_low = torch.FloatTensor(act_low)
        self._transforms = [
            SigmoidTransform(),
            AffineTransform(loc=act_low, scale=act_scale)
        ]

    def forward(self, x):
        dist = self.policy(x)
        action = dist.rsample() # Reparameterization trick
        logprobs = dist.log_prob(action)
        return action, logprobs

    def policy(self, x):
        x = F.relu(self._l1(x))
        x = F.relu(self._l2(x))
        mean = self._mean_layer(x)
        log_std = self._std_layer(x).clamp(self._log_std_min, self._log_std_max)
        std = torch.diag_embed(log_std.exp())
        base_distribution = MultivariateNormal(mean, std)
        return TransformedDistribution(base_distribution, self._transforms)

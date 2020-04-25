import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=256):
        super(ValueNetwork, self).__init__()
        self._l1 = nn.Linear(input_dim, hidden_size)
        self._l2 = nn.Linear(hidden_size, hidden_size)
        self._l3 = nn.Linear(hidden_size, 1)
        self._input_dim = input_dim

    def forward(self, x):
        x1 = F.relu(self._l1(x).clone())
        x2 = F.relu(self._l2(x1))
        return self._l3(x2)

    def exponential_smooth(self, other, tau):
        for (self_p, other_p) in zip(self.parameters(), other.parameters()):
            self_p.data.copy_(self_p.data * (1. - tau) + other_p.data * tau)

class CriticNetwork(ValueNetwork):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(CriticNetwork, self).__init__(obs_dim + act_dim, hidden_size=hidden_size)
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def forward(self, s, a):
        x = torch.cat([s.clone(), a.clone()], dim=1)
        return ValueNetwork.forward(self, x)

class DiscreteCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(DiscreteCriticNetwork, self).__init__()
        self._l1 = nn.Linear(obs_dim, hidden_size)
        self._l2 = nn.Linear(hidden_size, hidden_size)
        self._l3 = nn.Linear(hidden_size, act_dim)

    def forward(self, s, a):
        s1 = F.relu(self._l1(s).clone())
        s2 = F.relu(self._l2(s1))
        s3 = self._l3(s2)
        return s3.gather(1, a.long())

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self._obs_dim = obs_dim
        self._l1 = nn.Linear(obs_dim, hidden_size)
        self._l2 = nn.Linear(hidden_size, hidden_size)
        self._hidden_size = hidden_size

    def forward(self, x):
        x1 = F.relu(self._l1(x).clone())
        x2 = F.relu(self._l2(x1))
        return x2

    def policy(self, x):
        pass

class GaussianActorNetwork(ActorNetwork):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_low,
                 act_high,
                 log_std_min=-20,
                 log_std_max=20,
                 hidden_size=256):
        super(GaussianActorNetwork, self).__init__(obs_dim, hidden_size=hidden_size)
        self._mean_layer = nn.Linear(self._hidden_size, act_dim)
        self._std_layer = nn.Linear(self._hidden_size, act_dim)
        self._act_dim = act_dim
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        act_scale = torch.FloatTensor(act_high - act_low).to(device)
        act_low = torch.FloatTensor(act_low).to(device)
        self._transforms = [
            SigmoidTransform(),
            AffineTransform(loc=act_low, scale=act_scale)
        ]

    def forward(self, x):
        dist = self.policy(x.clone())
        action = dist.rsample() # Reparameterization trick
        logprobs = dist.log_prob(action)
        return action, logprobs

    def policy(self, x):
        logits = ActorNetwork.forward(self, x.clone())
        mean = self._mean_layer(logits)
        log_std = self._std_layer(logits).clamp(self._log_std_min, self._log_std_max)
        std = torch.diag_embed(log_std.exp())
        base_distribution = MultivariateNormal(mean, std)
        return TransformedDistribution(base_distribution, self._transforms)

class DiscreteActorNetwork(ActorNetwork):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 temperature=1e-5,
                 hidden_size=256):
        super(DiscreteActorNetwork, self).__init__(obs_dim, hidden_size=hidden_size)
        self._logits_layer = nn.Linear(self._hidden_size, act_dim)
        self._act_dim = act_dim
        self._temperature = temperature
        self._action_indices = torch.arange(act_dim).float().t()

    def forward(self, x):
        dist = self.policy(x.clone())
        one_hot_action = F.gumbel_softmax(dist.logits, tau=self._temperature, hard=True)
        action = one_hot_action @ self._action_indices
        logprobs = dist.log_prob(action)
        return action.unsqueeze(-1), logprobs

    def policy(self, x):
        x1 = ActorNetwork.forward(self, x.clone())
        logits = self._logits_layer(x1)
        return Categorical(logits=logits)

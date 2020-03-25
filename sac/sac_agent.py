import copy
import numpy as np
import torch
import torch.nn.functional as F

from sac.nets import ActorNetwork, CriticNetwork, ValueNetwork
from sac.replay import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SACAgent:
    def __init__(self,
                 env_fn,
                 pre_train_steps=1000,
                 max_replay_capacity=10000,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tau=5e-3,
                 gamma=0.99,
                 alpha=1.0,
                 batch_size=32,
                 value_delay=1):
        self._total_steps = 0
        self._env_fn = env_fn
        self._tau = tau
        self._gamma = gamma
        self._batch_size = batch_size
        self._value_delay = value_delay
        env = env_fn()

        self._pre_train_steps = pre_train_steps
        self._training = False

        self._obs_dim = np.prod(env.observation_space.shape)
        self._act_dim = np.prod(env.action_space.shape)
        act_high = env.action_space.high
        act_low = env.action_space.low

        self._replay_buf = ReplayBuffer(self._obs_dim, self._act_dim, max_replay_capacity)

        self._actor = ActorNetwork(self._obs_dim, self._act_dim, act_low, act_high).to(device)
        self._critic1 = CriticNetwork(self._obs_dim, self._act_dim).to(device)
        self._critic2 = CriticNetwork(self._obs_dim, self._act_dim).to(device)
        self._value_net = ValueNetwork(self._obs_dim).to(device)
        self._target_value_net = copy.deepcopy(self._value_net).to(device)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._value_optimizer = torch.optim.Adam(self._actor.parameters(), lr=critic_lr)
        self._critic1_optimizer = torch.optim.Adam(self._critic1.parameters(), lr=critic_lr)
        self._critic2_optimizer = torch.optim.Adam(self._critic2.parameters(), lr=critic_lr)

    def train(self, num_steps, visualizer=None):
        env = self._env_fn()
        s = env.reset()
        episode_reward = 0
        num_episodes = 0
        for i in range(num_steps):
            s_t = torch.FloatTensor(s).to(device).unsqueeze(0)
            if self._training:
                a, _ = self._actor(s_t)
                a = a.squeeze().detach().cpu().numpy()
            else:
                a = env.action_space.sample()
            ns, r, d, _ = env.step(a)
            episode_reward += r
            self._replay_buf.store(s, a, r, ns, d)
            self._total_steps += 1
            if not self._training:
                if self._total_steps >= self._pre_train_steps:
                    self._training = True
            if self._training:
                losses = self.update()
                artifacts = {'loss': losses,
                             'step': self._total_steps,
                             'episode': num_episodes,
                             'done': d,
                             'return': episode_reward}
                if visualizer is not None:
                    visualizer(artifacts)
            if d:
                s = env.reset()
                num_episodes += 1
                episode_reward = 0
            else:
                s = ns

    def update(self):
        s, a, r, ns, d = self._replay_buf.sample(self._batch_size)
        s = torch.FloatTensor(s).to(device)
        a = torch.FloatTensor(a).to(device)
        r = torch.FloatTensor(r).to(device)
        ns = torch.FloatTensor(ns).to(device)
        d = torch.FloatTensor(d).to(device)

        sampled_actions, log_probs = self._actor(s)
        sampled_q1 = self._critic1(s, sampled_actions).squeeze()
        sampled_q2 = self._critic2(s, sampled_actions).squeeze()
        sampled_q = torch.min(sampled_q1, sampled_q2)

        values = self._value_net(s)
        value_targets = sampled_q - log_probs
        value_loss = F.mse_loss(values, value_targets.detach())
        self._value_optimizer.zero_grad()
        value_loss.backward()
        self._value_optimizer.step()

        q1 = self._critic1(s, a)
        q2 = self._critic2(s, a)
        with torch.no_grad():
            target_q = r + (1.0 - d) * self._gamma * self._target_value_net(ns)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        self._critic1_optimizer.zero_grad()
        self._critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self._critic1_optimizer.step()
        self._critic2_optimizer.step()

        actor_loss = (log_probs - sampled_q).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        if self._total_steps % self._value_delay == 0:
            self._target_value_net.exponential_smooth(self._value_net, self._tau)

        return {
            'actor': actor_loss.item(),
            'critic1': critic1_loss.item(),
            'critic2': critic2_loss.item(),
            'value': value_loss.item(),
        }

    def action(self, x):
        x = torch.FloatTensor(x).to(device)
        action = self._actor(x)
        return action[0].squeeze().detach().cpu().numpy()

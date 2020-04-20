import copy
import gym
import numpy as np
import torch
import torch.nn.functional as F

from sac.nets import GaussianActorNetwork, DiscreteActorNetwork
from sac.nets import CriticNetwork, ValueNetwork
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
                 gamma=0.999,
                 alpha=1.0,
                 batch_size=32,
                 hidden_size=256,
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

        self._discrete_actions = False

        if isinstance(env.action_space, gym.spaces.Discrete):
            self._act_dim = env.action_space.n
            self._actor = DiscreteActorNetwork(self._obs_dim,
                                               self._act_dim,
                                               hidden_size=hidden_size).to(device)
            self._discrete_actions = True
        elif isinstance(env.action_space, gym.spaces.Box):
            self._act_dim = np.prod(env.action_space.shape)
            act_high = env.action_space.high
            act_low = env.action_space.low
            self._actor = GaussianActorNetwork(self._obs_dim,
                                               self._act_dim,
                                               act_low,
                                               act_high,
                                               hidden_size=hidden_size).to(device)
        else:
            raise NotImplementedError("Unsupported action space type: {}".format(env.action_space))

        self._replay_buf = ReplayBuffer(self._obs_dim, self._act_dim, max_replay_capacity)
        self._critic1 = CriticNetwork(self._obs_dim, self._act_dim, hidden_size=hidden_size).to(device)
        self._critic2 = CriticNetwork(self._obs_dim, self._act_dim, hidden_size=hidden_size).to(device)
        self._value_net = ValueNetwork(self._obs_dim, hidden_size=hidden_size).to(device)
        self._target_value_net = copy.deepcopy(self._value_net).to(device)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._value_optimizer = torch.optim.Adam(self._value_net.parameters(), lr=critic_lr)
        self._critic1_optimizer = torch.optim.Adam(self._critic1.parameters(), lr=critic_lr)
        self._critic2_optimizer = torch.optim.Adam(self._critic2.parameters(), lr=critic_lr)

    def rollout(self, num_rollouts):
        rewards = np.zeros(num_rollouts)
        for i in range(num_rollouts):
            env = self._env_fn()
            s = env.reset()
            episode_reward = 0
            done = False
            while not done:
                s_t = torch.from_numpy(s).float().to(device).unsqueeze(0)
                a, _ = self._actor(s_t)
                a = a.squeeze().detach().cpu().numpy()
                if self._discrete_actions:
                    a = int(a.item())
                s, r, done, _ = env.step(a)
                episode_reward += r
            rewards[i] = episode_reward
        return rewards

    def train(self, num_steps, win_condition=None, win_window=5, visualizer=None):
        env = self._env_fn()
        s = env.reset()
        episode_reward = 0
        num_episodes = 0
        if win_condition is not None:
            scores = [0. for _ in range(win_window)]
            idx = 0
        for i in range(num_steps):
            s_t = torch.from_numpy(s).float().to(device).unsqueeze(0)
            if self._training:
                a, _ = self._actor(s_t)
                a = a.squeeze().detach().cpu().numpy()
                if self._discrete_actions:
                    a = int(a.item())
            else:
                a = env.action_space.sample()
            ns, r, d, _ = env.step(a)
            episode_reward += r
            if self._discrete_actions:
                self._replay_buf.store(s, np.eye(self._act_dim)[int(a)], r, ns, d)
            else:
                self._replay_buf.store(s, a, r, ns, d)
            self._total_steps += 1
            if not self._training:
                if self._total_steps >= self._pre_train_steps:
                    self._training = True
            if self._training:
                losses = self.update()
                artifacts = {
                    'loss': losses,
                     'step': self._total_steps,
                     'episode': num_episodes,
                     'done': d,
                     'return': episode_reward,
                     'transition': {
                         'state': s,
                         'action': a,
                         'reward': r,
                         'next state': ns,
                         'done': d,
                     }
                }
                if visualizer is not None:
                    visualizer(artifacts)
            if d:
                s = env.reset()
                num_episodes += 1
                if win_condition is not None:
                    scores[idx] = episode_reward
                    idx = (idx + 1) % win_window
                    if (num_episodes >= win_window) and (np.mean(scores) >= win_condition):
                        print("SAC finished training: win condition reached")
                        break
                episode_reward = 0
            else:
                s = ns

    def update(self):
        s, a, r, ns, d = self._replay_buf.sample(self._batch_size)
        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a).float().to(device)
        r = torch.from_numpy(r).float().to(device)
        ns = torch.from_numpy(ns).float().to(device)
        d = torch.from_numpy(d).float().to(device)

        sampled_actions, log_probs = self._actor(s)
        if self._discrete_actions:
            sampled_actions = torch.eye(self._act_dim)[sampled_actions.long()]
        sampled_q1 = self._critic1(s, sampled_actions).squeeze()
        sampled_q2 = self._critic2(s, sampled_actions).squeeze()
        sampled_q = torch.min(sampled_q1, sampled_q2)

        values = (self._value_net(s)).squeeze()
        value_targets = (sampled_q - log_probs).squeeze()
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
        critic1_loss.backward()
        self._critic1_optimizer.step()
        self._critic2_optimizer.zero_grad()
        critic2_loss.backward()
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
        x = torch.from_numpy(x).float().to(device)
        action, _ = self._actor(x)
        action = action.squeeze().detach().cpu().numpy()
        if self._discrete_actions:
            return int(action.item())
        return action

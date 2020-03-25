import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_capacity=1000):
        self.capacity = max_capacity
        self._full = False
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._s_buf = np.zeros((max_capacity, obs_dim))
        self._a_buf = np.zeros((max_capacity, act_dim))
        self._r_buf = np.zeros((max_capacity, 1))
        self._d_buf = np.zeros((max_capacity, 1))
        self._ns_buf = np.zeros((max_capacity, obs_dim))
        self._ptr = 0

    def store(self, s, a, r, ns, d):
        self._s_buf[self._ptr] = s
        self._a_buf[self._ptr] = a
        self._r_buf[self._ptr] = r
        self._d_buf[self._ptr] = float(d)
        self._ns_buf[self._ptr] = ns
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0:
            self._full = True

    def sample(self, batch_size):
        if self._full:
            indices = np.random.choice(self.capacity, size=batch_size)
        else:
            indices = np.random.choice(self._ptr, size=batch_size)
        s_batch = self._s_buf[indices]
        a_batch = self._a_buf[indices]
        r_batch = self._r_buf[indices]
        ns_batch = self._ns_buf[indices]
        d_batch = self._d_buf[indices]
        return s_batch, a_batch, r_batch, ns_batch, d_batch

    def is_full(self):
        return self._full

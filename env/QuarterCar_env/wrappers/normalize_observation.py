import numpy as np
import gymnasium as gym
from pathlib import Path


class NormalizeObservation(gym.ObservationWrapper):
    """Running mean/std normalisation with save/load."""

    def __init__(self, env, epsilon: float = 1e-8, stats_path: str = None):
        super().__init__(env)
        self.epsilon = epsilon
        self.stats_path = stats_path
        n = env.observation_space.shape[0]
        self._mean  = np.zeros(n, dtype=np.float64)
        self._var   = np.ones(n,  dtype=np.float64)
        self._count = 0
        if stats_path and Path(stats_path).exists():
            self.load_stats(stats_path)

    def observation(self, obs):
        self._count += 1
        delta       = obs - self._mean
        self._mean += delta / self._count
        self._var  += delta * (obs - self._mean)
        std = np.sqrt(self._var / max(self._count, 1) + self.epsilon)
        return ((obs - self._mean) / std).astype(np.float32)

    def save_stats(self, path: str):
        np.savez(path, mean=self._mean, var=self._var, count=np.array(self._count))

    def load_stats(self, path: str):
        d = np.load(path)
        self._mean, self._var, self._count = d['mean'], d['var'], int(d['count'])

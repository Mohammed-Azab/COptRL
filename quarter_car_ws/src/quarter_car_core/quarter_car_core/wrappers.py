"""Gymnasium wrappers for the quarter-car environment."""
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym


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


class ActionRepeat(gym.Wrapper):
    """Repeat action n_repeat times, accumulate reward."""

    def __init__(self, env, n_repeat: int = 2):
        super().__init__(env)
        self.n_repeat = n_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.n_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class RewardScaler(gym.RewardWrapper):
    """Scale reward by a constant."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


class EpisodeLogger(gym.Wrapper):
    """Append one CSV row per episode: episode, return, rms_accel, peak_accel, road_profile."""

    def __init__(self, env, log_dir: str):
        super().__init__(env)
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._csv_path  = os.path.join(log_dir, f'episodes_{ts}.csv')
        self._episode   = 0
        self._ep_return = 0.0
        with open(self._csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['episode', 'return', 'rms_accel', 'peak_accel', 'road_profile'])

    def reset(self, **kwargs):
        self._ep_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += reward
        if terminated or truncated:
            self._episode += 1
            with open(self._csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self._episode,
                    round(self._ep_return, 4),
                    round(info.get('rms_accel', 0.0), 4),
                    round(info.get('peak_accel', 0.0), 4),
                    info.get('road_profile', ''),
                ])
        return obs, reward, terminated, truncated, info

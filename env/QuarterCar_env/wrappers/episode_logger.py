import csv
import os
from datetime import datetime

import gymnasium as gym


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

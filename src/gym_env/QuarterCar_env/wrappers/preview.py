from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import find_peaks

from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config
from QuarterCar_env.config.env_params import DT

_DENSE_N = 200   # spatial samples for peak detection over the preview horizon


class PreviewWrapper(gym.ObservationWrapper):
    # appends [dist, height, width] × n_peaks to the base observation

    def __init__(self, env: gym.Env, cfg: RewardConfig | None = None):
        super().__init__(env)

        self._cfg = cfg or load_reward_config()
        n_prev    = self._cfg.n_peaks * 3

        orig_high = env.observation_space.high
        orig_low  = env.observation_space.low

        # peak slots: dist and width in [0,1], height in [0,1]
        self.observation_space = spaces.Box(
            low  = np.concatenate([orig_low,  np.zeros(n_prev, dtype=np.float32)]),
            high = np.concatenate([orig_high, np.ones(n_prev,  dtype=np.float32)]),
            dtype=np.float32,
        )

        # PT1 filter state — reset every episode
        self.preview: np.ndarray = np.tile(
            [1.0, 0.0, 0.0], self._cfg.n_peaks
        ).astype(np.float32)
        self._filtered_preview = self.preview.copy()

    # gymnasium hook — called after every step and reset
    def observation(self, obs: np.ndarray) -> np.ndarray:
        peak_obs = self._compute_peaks()
        return np.concatenate([obs, peak_obs]).astype(np.float32)

    def reset(self, **kwargs):
        self._filtered_preview = np.tile(
            [1.0, 0.0, 0.0], self._cfg.n_peaks
        ).astype(np.float32)
        self.preview = self._filtered_preview.copy()
        return super().reset(**kwargs)

    def _compute_peaks(self) -> np.ndarray:
        cfg  = self._cfg
        env  = self.env.unwrapped

        heights = env._road.get_spatial_preview(
            s_pos=env._s_pos,
            t_current=env._t,
            v_current=max(env._v, 0.5),
            lookahead_m=cfg.preview_distance,
            n_points=_DENSE_N,
        )

        ds               = cfg.preview_distance / _DENSE_N
        min_dist_samples = max(1, int(cfg.peak_distance_min_m / ds))

        peaks, props = find_peaks(
            heights,
            height=cfg.peak_height_min,
            distance=min_dist_samples,
            width=0,
        )

        # default: bump at the horizon, zero height/width
        peak_arr = np.tile([1.0, 0.0, 0.0], cfg.n_peaks).astype(np.float32)

        for i, pk in enumerate(peaks[: cfg.n_peaks]):
            peak_arr[i * 3]     = float(pk * ds / cfg.preview_distance)
            peak_arr[i * 3 + 1] = float(np.clip(heights[pk] / cfg.h_clip, 0.0, 1.0))
            peak_arr[i * 3 + 2] = float(
                np.clip(props["widths"][i] * ds / cfg.preview_distance, 0.0, 1.0)
            )

        # raw peaks (pre-noise) available via get_wrapper_attr("preview")
        self.preview = peak_arr.copy()

        if cfg.noise_active and self.np_random is not None:
            rng = self.np_random
            for i in range(cfg.n_peaks):
                if peak_arr[i * 3] < 1.0:
                    scale = peak_arr[i * 3]
                    peak_arr[i * 3]     += float(rng.normal(0, cfg.noise_distance_std)) * scale
                    peak_arr[i * 3 + 1] += float(rng.normal(0, cfg.noise_height_std))   * scale
                    peak_arr[i * 3 + 2] += float(rng.normal(0, cfg.noise_width_std))    * scale
            peak_arr = np.clip(peak_arr, 0.0, 1.0)

        alpha = DT / (cfg.pt1_tau + DT)
        self._filtered_preview = (
            self._filtered_preview + alpha * (peak_arr - self._filtered_preview)
        )

        return np.clip(self._filtered_preview, 0.0, 1.0)

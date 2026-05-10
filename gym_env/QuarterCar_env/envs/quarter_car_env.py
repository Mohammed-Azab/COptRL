"""
QuarterCarEnv Gymnasium environment for quarter-car active suspension RL.

Observation (8,) float32:
  idx 0: z_s              [m]     clipped +-0.5
  idx 1: z_s_dot          [m/s]   clipped +-5
  idx 2: z_u              [m]     clipped +-0.5
  idx 3: z_u_dot          [m/s]   clipped +-5
  idx 4: z_r              [m]     clipped +-0.2
  idx 5: z_r_dot          [m/s]   clipped +-2
  idx 6: suspension_travel = z_s - z_u  [m]  clipped +-0.15
  idx 7: tyre_deflection   = z_u - z_r  [m]  clipped +-0.1

Action (1,) float32 in [-1, 1]:
  F_act = action[0] * F_MAX   (F_MAX = 10000 N)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuarterCar_env.ode_model import QuarterCarODE
from QuarterCar_env.road_generator import RoadGenerator
from QuarterCar_env.reward import RewardConfig, compute_reward, compute_terminal_bonus
from QuarterCar_env.params import (
    F_MAX, DT, EPISODE_STEPS,
    TRUNC_TRAVEL, TRUNC_ZS,
    OBS_HIGH, OBS_LOW,
    VEHICLE_SPEED,
)


class QuarterCarEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 50}

    def __init__(
        self,
        road_profile: str = 'iso_8608_class_c',
        vehicle_speed: float = VEHICLE_SPEED,
        render_mode: str = 'none',
        physics_params: dict = None,
        road_params: dict = None,
        reward_config: RewardConfig = None,
    ):
        super().__init__()
        self.render_mode  = render_mode
        self.road_profile = road_profile

        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._ode  = QuarterCarODE(physics_params)
        self._road = RoadGenerator(road_profile, vehicle_speed, road_params)
        self._rcfg = reward_config or RewardConfig()

        self._state      = np.zeros(4, dtype=np.float64)
        self._t          = 0.0
        self._step_count = 0
        self._accel_sq   = 0.0
        self._peak_accel = 0.0
        self._travel_sq  = 0.0
        self._last_F     = 0.0
        self._hist       = None
        self._fig        = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self._ode.reset()
        self._road.reset(seed=int(rng.integers(0, 2**31)))

        self._state      = rng.normal(0.0, 0.005, size=4)
        self._t          = 0.0
        self._step_count = 0
        self._accel_sq   = 0.0
        self._peak_accel = 0.0
        self._travel_sq  = 0.0
        self._last_F     = 0.0
        self._hist       = None

        return self._obs(), self._info(0.0)

    def step(self, action):
        F_act = float(np.clip(action[0], -1.0, 1.0)) * F_MAX
        self._last_F = F_act

        z_r     = self._road.get_height(self._t)
        z_r_dot = self._road.get_height_dot(self._t)

        new_state, z_s_ddot = self._ode.step(self._state, F_act, z_r, DT)
        self._state = new_state
        self._t    += DT
        self._step_count += 1

        z_s, _, z_u, _ = self._state
        travel = z_s - z_u
        tyre   = z_u - z_r

        self._accel_sq   += z_s_ddot ** 2
        self._travel_sq  += travel ** 2
        self._peak_accel  = max(self._peak_accel, abs(z_s_ddot))

        reward = compute_reward(z_s_ddot, travel, tyre, F_act, self._rcfg)

        # gymnasium identity-checks `is False`, so we must return Python bool, not numpy bool
        truncated  = bool(abs(travel) > TRUNC_TRAVEL or abs(z_s) > TRUNC_ZS)
        terminated = False

        if self._step_count >= EPISODE_STEPS and not truncated:
            terminated = True
            rms = np.sqrt(self._accel_sq / self._step_count)
            reward += compute_terminal_bonus(rms, self._rcfg)

        if self.render_mode == 'human':
            self._do_render(z_r, F_act)

        return self._obs(), reward, terminated, truncated, self._info(z_s_ddot)

    def _obs(self) -> np.ndarray:
        z_s, z_s_dot, z_u, z_u_dot = self._state
        z_r     = self._road.get_height(self._t)
        z_r_dot = self._road.get_height_dot(self._t)
        raw = np.array([
            z_s, z_s_dot, z_u, z_u_dot,
            z_r, z_r_dot,
            z_s - z_u,
            z_u - z_r,
        ], dtype=np.float32)
        return np.clip(raw, OBS_LOW, OBS_HIGH)

    def _info(self, z_s_ddot: float) -> dict:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return {
            'rms_accel':      float(rms),
            'peak_accel':     float(self._peak_accel),
            'suspension_rms': float(np.sqrt(self._travel_sq / n)),
            'comfort_score':  float(max(0.0, 1.0 - rms / self._rcfg.a_limit)),
            'road_profile':   self.road_profile,
            'step_count':     self._step_count,
            'episode_time':   self._t,
        }

    def render(self):
        if self.render_mode == 'none':
            return None
        return self._do_render(self._road.get_height(self._t), self._last_F)

    def _do_render(self, z_r: float, F_act: float):
        if self._hist is None:
            self._hist = {'t': [], 'z_s': [], 'z_u': [], 'z_r': [], 'F': []}
        self._hist['t'].append(self._t)
        self._hist['z_s'].append(float(self._state[0]))
        self._hist['z_u'].append(float(self._state[2]))
        self._hist['z_r'].append(z_r)
        self._hist['F'].append(F_act)
        if self.render_mode == 'human':
            return self._render_human()
        return self._render_rgb_array()

    def _render_human(self):
        import os
        import matplotlib
        if not os.environ.get('DISPLAY'):
            matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._axes = plt.subplots(2, 1, figsize=(10, 5))
            plt.ion()
        ax1, ax2 = self._axes
        ax1.cla()
        ax1.plot(self._hist['t'], self._hist['z_s'], label='z_s (sprung)')
        ax1.plot(self._hist['t'], self._hist['z_u'], label='z_u (unsprung)')
        ax1.plot(self._hist['t'], self._hist['z_r'], '--', label='z_r (road)')
        ax1.legend(fontsize=8)
        ax1.set_ylabel('Height [m]')
        ax2.cla()
        ax2.plot(self._hist['t'], self._hist['F'])
        ax2.set_ylabel('F_act [N]')
        ax2.set_xlabel('Time [s]')
        plt.tight_layout()
        plt.pause(0.001)
        return None

    def _render_rgb_array(self):
        import io
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        axes[0].plot(self._hist['t'], self._hist['z_s'], label='z_s')
        axes[0].plot(self._hist['t'], self._hist['z_u'], label='z_u')
        axes[0].plot(self._hist['t'], self._hist['z_r'], '--', label='z_r')
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel('Height [m]')
        axes[1].plot(self._hist['t'], self._hist['F'])
        axes[1].set_ylabel('F_act [N]')
        axes[1].set_xlabel('Time [s]')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=72)
        buf.seek(0)
        from PIL import Image
        img = np.array(Image.open(buf).convert('RGB'))
        plt.close(fig)
        return img

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    def get_comfort_metric(self) -> float:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return float(max(0.0, 1.0 - rms / self._rcfg.a_limit))


""" QuarterCarEnv: A Gymnasium environment for quarter-car active suspension RL. """

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuarterCar_env.ode_model import MandlQuarterCarODE
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
        self._v0          = float(vehicle_speed)

        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0],  dtype=np.float32),
            dtype=np.float32,
        )

        self._ode  = MandlQuarterCarODE(physics_params)
        self._road = RoadGenerator(road_profile, vehicle_speed, road_params)
        self._rcfg = reward_config or RewardConfig()

        self._state      = self._ode.reset(self._v0)
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

        self._road.reset(seed=int(rng.integers(0, 2**31)))

        x = self._ode.reset(self._v0)
        # small random perturbation on vertical states
        x[0:4] += rng.normal(0.0, 0.005, size=4)
        x[5]   += rng.normal(0.0, 0.001)
        self._state = x

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

        new_state, z_B_ddot = self._ode.step(
            self._state, self._road.get_height_dot, self._t, F_act
        )
        self._state = new_state
        self._t    += DT
        self._step_count += 1

        # Mandl state aliases
        tyre_defl = float(new_state[0])   # ζ − z_W
        travel    = float(new_state[2])   # z_W − z_B

        self._accel_sq   += z_B_ddot ** 2
        self._travel_sq  += travel ** 2
        self._peak_accel  = max(self._peak_accel, abs(z_B_ddot))

        reward = compute_reward(z_B_ddot, travel, tyre_defl, F_act, self._rcfg)

        truncated  = bool(abs(travel) > TRUNC_TRAVEL or abs(float(new_state[5])) > TRUNC_ZS)
        terminated = False

        if self._step_count >= EPISODE_STEPS and not truncated:
            terminated = True
            rms = np.sqrt(self._accel_sq / self._step_count)
            reward += compute_terminal_bonus(rms, self._rcfg)

        if self.render_mode == 'human':
            self._do_render(self._road.get_height(self._t), F_act)

        return self._obs(), reward, terminated, truncated, self._info(z_B_ddot)

    def _obs(self) -> np.ndarray:
        x   = self._state
        zeta     = self._road.get_height(self._t)
        zeta_dot = self._road.get_height_dot(self._t)
        z_B  = float(x[5])
        z_W  = z_B + float(x[2])   # z_B + (z_W − z_B)
        raw = np.array([
            z_B,          # idx 0: body displacement
            float(x[3]),  # idx 1: body velocity
            z_W,          # idx 2: wheel displacement
            float(x[1]),  # idx 3: wheel velocity
            zeta,         # idx 4: road height
            zeta_dot,     # idx 5: road velocity
            float(x[2]),  # idx 6: suspension travel
            float(x[0]),  # idx 7: tire deflection
        ], dtype=np.float32)
        return np.clip(raw, OBS_LOW, OBS_HIGH)

    def _info(self, z_B_ddot: float) -> dict:
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

    def _do_render(self, zeta: float, F_act: float):
        if self._hist is None:
            self._hist = {'t': [], 'z_B': [], 'z_W': [], 'zeta': [], 'F': []}
        z_B = float(self._state[5])
        z_W = z_B + float(self._state[2])
        self._hist['t'].append(self._t)
        self._hist['z_B'].append(z_B)
        self._hist['z_W'].append(z_W)
        self._hist['zeta'].append(zeta)
        self._hist['F'].append(F_act)
        if self.render_mode == 'human':
            return self._render_human()
        return self._render_rgb_array()

    def _render_human(self):
        import os, matplotlib
        if not os.environ.get('DISPLAY'):
            matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._axes = plt.subplots(2, 1, figsize=(10, 5))
            plt.ion()
        ax1, ax2 = self._axes
        ax1.cla()
        ax1.plot(self._hist['t'], self._hist['z_B'],   label='z_B (body)')
        ax1.plot(self._hist['t'], self._hist['z_W'],   label='z_W (wheel)')
        ax1.plot(self._hist['t'], self._hist['zeta'], '--', label='ζ (road)')
        ax1.legend(fontsize=8); ax1.set_ylabel('Displacement [m]')
        ax2.cla()
        ax2.plot(self._hist['t'], self._hist['F'])
        ax2.set_ylabel('F_act [N]'); ax2.set_xlabel('Time [s]')
        plt.tight_layout(); plt.pause(0.001)
        return None

    def _render_rgb_array(self):
        import io, matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        axes[0].plot(self._hist['t'], self._hist['z_B'],   label='z_B')
        axes[0].plot(self._hist['t'], self._hist['z_W'],   label='z_W')
        axes[0].plot(self._hist['t'], self._hist['zeta'], '--', label='ζ')
        axes[0].legend(fontsize=8); axes[0].set_ylabel('Displacement [m]')
        axes[1].plot(self._hist['t'], self._hist['F'])
        axes[1].set_ylabel('F_act [N]'); axes[1].set_xlabel('Time [s]')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=72)
        buf.seek(0)
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

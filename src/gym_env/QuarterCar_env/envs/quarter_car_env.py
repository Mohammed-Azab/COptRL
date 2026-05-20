"""
QuarterCarEnv: Gymnasium speed-planning environment (quarter-car, 6-state ODE).

Action (1,) float32 in [-1, 1]:
  u = action[0]
  a_cmd = u * a_max   [m/s^2]
  v_{t+1} = clip(v_t + a_cmd * DT, 0, v_max)

State (internal, 6-D float64):
  x[0] = zeta - z_W   tyre deflection              [m]
  x[1] = dz_W         wheel vertical velocity       [m/s]
  x[2] = z_W - z_B    suspension travel             [m]
  x[3] = dz_B         body vertical velocity        [m/s]
  x[4] = v            longitudinal speed            [m/s]
  x[5] = z_B          body displacement from eq.    [m]

Observation (float32) indices depend on obs_enable_* flags in RewardConfig:
  0: z_B              body displacement             [m]
  1: dz_B             body velocity                 [m/s]
  2: z_W              wheel displacement            [m]
  3: dz_W             wheel velocity                [m/s]
  4: zeta             road height                   [m]
  5: dzeta            road velocity                 [m/s]
  6: z_W - z_B        suspension travel             [m]
  7: zeta - z_W       tyre deflection               [m]
  8: v / v_max        normalised speed              [-]
  9: (v_target-v)/v_max  normalised speed error     [-]
  10 (if obs_enable_accel):       filtered_a / a_comfort  [-]
  11 (if obs_enable_jerk):        filtered_jerk / j_max   [-]
  12 (if obs_enable_prev_action): prev_action             [-]
  13 (if obs_enable_curvature):   curvature / curvature_clip  [-]
"""

from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuarterCar_env.core.ode_model import QuarterCarODE
from road.road_generator import RoadGenerator
from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config
from QuarterCar_env.reward.reward import compute_reward, compute_v_target, compute_terminal_bonus
from QuarterCar_env.config.env_params import (
    DT, EPISODE_STEPS,
    TRUNC_TRAVEL, TRUNC_ZS, MAX_DISTANCE,
    OBS_HIGH, OBS_LOW,
)
from QuarterCar_env.config.road_params import VEHICLE_SPEED, V_BRAKE_LEAD
from QuarterCar_env.config.render_params import (
    RENDER_Y_SCALE,
    RENDER_SHOW_TS,
    RENDER_TS_Z,
    RENDER_TS_Z_DDOT,
    RENDER_TS_SPEED,
    RENDER_FREEZE_EPISODE,
)
from .render import render_env, close_env


class QuarterCarEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'none'],
        'render_fps': int(round(1.0 / DT)),
    }

    def __init__(
        self,
        road_profile: str = 'iso_8608_class_c',
        vehicle_speed: float = VEHICLE_SPEED,
        render_mode: str = 'none',
        physics_params: dict = None,
        road_params: dict = None,
        reward_config: RewardConfig = None,
        render_y_scale: int = RENDER_Y_SCALE,
        show_time_series: bool = RENDER_SHOW_TS,
        show_ts_z: bool = RENDER_TS_Z,
        show_ts_z_ddot: bool = RENDER_TS_Z_DDOT,
        show_ts_speed: bool = RENDER_TS_SPEED,
        freeze_episode: bool = RENDER_FREEZE_EPISODE,
        start_at_equilibrium: bool = True,
        ref_speed_profile: str = "constant",    # constant | slow_before_bump | custom
        max_episode_steps: int = EPISODE_STEPS,
        max_distance: Optional[float] = MAX_DISTANCE,
    ):
        super().__init__()
        self.render_mode        = render_mode
        self.road_profile       = road_profile
        self._v0                = float(vehicle_speed)
        self._y_scale           = int(render_y_scale)
        self._ref_speed_profile = ref_speed_profile
        self._max_episode_steps = int(max_episode_steps)
        self._max_distance      = max_distance
        self._start_at_eq       = bool(start_at_equilibrium)

        self._v_ref_fn: Optional[Callable[[float], float]] = (road_params or {}).get('v_ref_fn', None)

        self._rcfg = reward_config or load_reward_config()

        # action space: normalised acceleration command
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
        )

        # observation space — shape determined once from toggle flags
        obs_high, obs_low = self._build_obs_bounds()
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._ode  = QuarterCarODE(physics_params)
        self._road = RoadGenerator(road_profile, vehicle_speed, road_params)

        # episode state
        self._state          = self._ode.reset(self._v0)
        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._travel_sq      = 0.0
        self._last_z_B_ddot  = 0.0
        self._episode_reward = 0.0

        # speed state
        self._v              = self._v0
        self._v_ref_last     = self._rcfg.v_max
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        # filter state — cleared in reset()
        self._prev_a         = 0.0
        self._filtered_a     = 0.0
        self._filtered_jerk  = 0.0
        self._prev_action    = 0.0
        self._curvature      = 0.0

        self._bump_times     = self._road.get_bump_times()
        self._fig            = None
        self._ren_hist       = None
        self._episode_count  = 0
        self._show_ts        = bool(show_time_series)
        self._freeze_episode = bool(freeze_episode)
        self._freeze_render  = False
        self._ts_flags = {
            "z": bool(show_ts_z),
            "z_ddot": bool(show_ts_z_ddot),
            "speed": bool(show_ts_speed),
        }

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self._road.reset(seed=int(rng.integers(0, 2**31)))
        self._road.set_speed(self._v0)

        x = self._ode.reset(self._v0)
        if not self._start_at_eq:
            x[0:4] += rng.normal(0.0, 0.005, size=4)
            x[5]   += rng.normal(0.0, 0.001)
        self._state = x

        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._travel_sq      = 0.0
        self._last_z_B_ddot  = 0.0
        self._episode_reward = 0.0
        self._episode_count += 1

        if self._fig is not None:
            close_env(self)
        self._ren_hist = None

        self._v              = self._v0
        self._v_ref_last     = self._rcfg.v_max
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        # reset filter state
        self._prev_a         = 0.0
        self._filtered_a     = 0.0
        self._filtered_jerk  = 0.0
        self._prev_action    = 0.0

        self._bump_times = self._road.get_bump_times()

        self._freeze_render = False

        return self._obs(), self._info(0.0)

    def step(self, action):
        cfg = self._rcfg

        # 1. Acceleration command -> speed integration
        u     = float(np.clip(action[0], -1.0, 1.0))
        a_cmd = u * cfg.a_max
        v_old = self._v
        v_new = float(np.clip(v_old + a_cmd * DT, 0.0, cfg.v_max))
        a_actual = (v_new - v_old) / DT
        self._v        = v_new
        self._state[4] = v_new
        self._road.set_speed(v_new)
        self._s_pos   += v_new * DT

        # 2. Update IIR filters
        alpha_a = cfg.accel_filter_alpha
        a_clipped = float(np.clip(a_actual, -cfg.accel_clip, cfg.accel_clip))
        self._filtered_a = alpha_a * self._filtered_a + (1.0 - alpha_a) * a_clipped

        jerk = (a_actual - self._prev_a) / DT
        alpha_j = cfg.jerk_filter_alpha
        j_clipped = float(np.clip(jerk, -cfg.jerk_clip, cfg.jerk_clip))
        self._filtered_jerk = alpha_j * self._filtered_jerk + (1.0 - alpha_j) * j_clipped

        # 3. Integrate ODE one control step
        new_state, z_B_ddot = self._ode.step(
            self._state, self._road.get_height_dot, self._t
        )
        self._state         = new_state
        self._t            += DT
        self._step_count   += 1
        self._last_z_B_ddot = z_B_ddot

        travel = float(new_state[2])

        self._accel_sq  += z_B_ddot ** 2
        self._travel_sq += travel ** 2
        self._peak_accel = max(self._peak_accel, abs(z_B_ddot))

        # 4. Speed reference and reward
        v_ref    = self._compute_v_ref(self._t)
        v_target = compute_v_target(v_ref, cfg.target_speed_mode, self._curvature, cfg)
        self._v_ref_last = v_target

        reward, breakdown = compute_reward(
            v_new, v_target,
            a_actual, self._filtered_a,
            jerk, self._filtered_jerk,
            self._prev_action, u,
            self._curvature, cfg,
        )
        self._speed_err_sq   += (v_target - v_new) ** 2
        self._episode_reward += reward

        # 5. Update history
        self._prev_a      = a_actual
        self._prev_action = u

        # 6. Termination / truncation
        truncated = bool(
            abs(travel) > TRUNC_TRAVEL
            or abs(float(new_state[5])) > TRUNC_ZS
            or (self._max_distance is not None and self._s_pos >= self._max_distance)
        )
        terminated = False

        if self._step_count >= self._max_episode_steps and not truncated:
            terminated = True
            rms = np.sqrt(self._accel_sq / self._step_count)
            reward += compute_terminal_bonus(rms, cfg)

        if self.render_mode == 'human':
            self.render()

        if (terminated or truncated) and self.render_mode == 'human' and self._freeze_episode:
            self._freeze_render = True
            self.render()

        info = self._info(z_B_ddot)
        info.update(breakdown)
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return render_env(self)

    def close(self):
        close_env(self)

    # ------------------------------------------------------------------
    # External curvature input
    # ------------------------------------------------------------------

    def set_curvature(self, k: float) -> None:
        """Set road curvature [m^-1] from an external planner. Call before step()."""
        self._curvature = float(k)

    # ------------------------------------------------------------------
    # Observation / info helpers
    # ------------------------------------------------------------------

    def _build_obs_bounds(self):
        """Build observation space bounds once at __init__ from toggle flags."""
        cfg = self._rcfg

        # Speed components always present (values are normalised by v_max)
        extra_high = [1.0,  1.0]
        extra_low  = [0.0, -1.0]

        if cfg.obs_enable_accel:
            bound = cfg.accel_clip / cfg.a_comfort
            extra_high.append(bound)
            extra_low.append(-bound)
        if cfg.obs_enable_jerk:
            bound = cfg.jerk_clip / cfg.j_max
            extra_high.append(bound)
            extra_low.append(-bound)
        if cfg.obs_enable_prev_action:
            extra_high.append(1.0)
            extra_low.append(-1.0)
        if cfg.obs_enable_curvature:
            extra_high.append(1.0)
            extra_low.append(-1.0)

        high = np.concatenate([OBS_HIGH, extra_high]).astype(np.float32)
        low  = np.concatenate([OBS_LOW,  extra_low ]).astype(np.float32)
        return high, low

    def _obs(self) -> np.ndarray:
        x        = self._state
        zeta     = self._road.get_height(self._t)
        zeta_dot = self._road.get_height_dot(self._t)
        z_B = float(x[5])
        z_W = z_B + float(x[2])

        raw = np.array([
            z_B, float(x[3]), z_W, float(x[1]),
            zeta, zeta_dot, float(x[2]), float(x[0]),
        ], dtype=np.float32)
        base_obs = np.clip(raw, OBS_LOW, OBS_HIGH)

        cfg = self._rcfg
        v     = self._v
        v_ref = self._v_ref_last

        extras = [
            float(np.clip(v / cfg.v_max,           0.0,  1.0)),
            float(np.clip((v_ref - v) / cfg.v_max, -1.0, 1.0)),
        ]
        if cfg.obs_enable_accel:
            bound = cfg.accel_clip / cfg.a_comfort
            extras.append(float(np.clip(self._filtered_a / cfg.a_comfort, -bound, bound)))
        if cfg.obs_enable_jerk:
            bound = cfg.jerk_clip / cfg.j_max
            extras.append(float(np.clip(self._filtered_jerk / cfg.j_max, -bound, bound)))
        if cfg.obs_enable_prev_action:
            extras.append(float(self._prev_action))
        if cfg.obs_enable_curvature:
            k_norm = np.clip(self._curvature, -cfg.curvature_clip, cfg.curvature_clip) / cfg.curvature_clip
            extras.append(float(k_norm))

        return np.concatenate([base_obs, extras]).astype(np.float32)

    def _info(self, z_B_ddot: float) -> dict:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return {
            'episode_reward': float(self._episode_reward),
            'rms_accel':       float(rms),
            'peak_accel':      float(self._peak_accel),
            'suspension_rms':  float(np.sqrt(self._travel_sq / n)),
            'comfort_score':   float(max(0.0, 1.0 - rms / self._rcfg.a_limit)),
            'road_profile':    self.road_profile,
            'step_count':      self._step_count,
            'episode_time':    self._t,
            'z_B_ddot':        float(z_B_ddot),
            'speed':           float(self._v),
            'v_ref':           float(self._v_ref_last),
            'speed_error':     float(self._v_ref_last - self._v),
            'speed_error_rms': float(np.sqrt(self._speed_err_sq / n)),
        }

    def get_comfort_metric(self) -> float:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return float(max(0.0, 1.0 - rms / self._rcfg.a_limit))

    # ------------------------------------------------------------------
    # Speed reference profile
    # ------------------------------------------------------------------

    def _compute_v_ref(self, t: float) -> float:
        v_max = self._rcfg.v_max
        v_min = self._rcfg.min_curve_speed
        if self._ref_speed_profile == "constant":
            return v_max
        if self._ref_speed_profile == "custom":
            return float(self._v_ref_fn(t))
        if self._ref_speed_profile == "slow_before_bump":
            times = self._bump_times
            if not times:
                return v_max
            t_start, t_center, t_end = times
            t_brake_start = t_center - V_BRAKE_LEAD
            t_accel_end   = t_end    + V_BRAKE_LEAD
            if t < t_brake_start:
                return v_max
            if t < t_center:
                alpha = (t - t_brake_start) / V_BRAKE_LEAD
                return v_max - (v_max - v_min) * alpha
            if t <= t_end:
                return v_min
            if t <= t_accel_end:
                alpha = (t - t_end) / V_BRAKE_LEAD
                return v_min + (v_max - v_min) * alpha
            return v_max
        return v_max

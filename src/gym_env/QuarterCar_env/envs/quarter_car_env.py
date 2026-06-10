from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuarterCar_env.core.ode_model import QuarterCarODE
from road.road_generator import RoadGenerator
from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config
from QuarterCar_env.reward.reward import compute_reward, compute_terminal_bonus
from QuarterCar_env.config.env_params import (
    DT, EPISODE_STEPS,
    TRUNC_TRAVEL, TRUNC_ZS,
    OBS_HIGH, OBS_LOW,
)
from QuarterCar_env.config.road_params import VEHICLE_SPEED
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
        physics_params: Optional[dict] = None,
        road_params: Optional[dict] = None,
        reward_config: Optional[RewardConfig] = None,
        render_y_scale: int = RENDER_Y_SCALE,
        show_time_series: bool = RENDER_SHOW_TS,
        show_ts_z: bool = RENDER_TS_Z,
        show_ts_z_ddot: bool = RENDER_TS_Z_DDOT,
        show_ts_speed: bool = RENDER_TS_SPEED,
        freeze_episode: bool = RENDER_FREEZE_EPISODE,
        start_at_equilibrium: bool = True,
        ref_speed_profile: str = "constant",    # constant | custom
        max_episode_steps: int = EPISODE_STEPS,
        max_distance: Optional[float] = None,
        trunc_travel: float = TRUNC_TRAVEL,
        trunc_zs: float = TRUNC_ZS,
        random_road_on_reset: bool = True,
        road_override_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.render_mode        = render_mode
        self.road_profile       = road_profile
        self._v0                = float(vehicle_speed)
        self._random_road_on_reset = bool(random_road_on_reset)
        self._y_scale           = int(render_y_scale)
        self._ref_speed_profile = ref_speed_profile
        self._max_episode_steps = int(max_episode_steps)
        self._trunc_travel      = float(trunc_travel)
        self._trunc_zs          = float(trunc_zs)
        self._start_at_eq       = bool(start_at_equilibrium)

        self._v_ref_fn: Optional[Callable[[float], float]] = (road_params or {}).get('v_ref_fn', None)

        self._rcfg = reward_config or load_reward_config()

        # action space: normalised acceleration command
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
        )

        # observation space
        obs_high, obs_low = self._build_obs_bounds()
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._ode  = QuarterCarODE(physics_params)
        self._road = RoadGenerator(road_profile, vehicle_speed, road_params)

        self._max_distance = (
            float(max_distance) if max_distance is not None
            else self._compute_max_distance()
        )

        # episode state
        self._state          = self._ode.reset(self._v0)
        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._last_z_B_ddot  = 0.0
        self._last_z_W_ddot  = 0.0
        self._episode_reward = 0.0

        # speed state
        self._v              = self._v0
        self._v_ref_last     = self._rcfg.v_max
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        # bump-crossing state
        self._bumps_passed   = 0
        self._bump_ends: list = []

        # random road config — read once, forwarded to from_random each reset
        from QuarterCar_env.config.config_manager import _load_yaml
        _rd_cfg = _load_yaml("road_params.yaml").get("random", {})
        self._random_road_kwargs = {
            "num_bumps_range": tuple(_rd_cfg.get("num_bumps_range", [1, 5])),
            "catalog_ids":     list(_rd_cfg.get("catalog_ids",      [0, 1, 2, 3, 4])),
            "min_gap":         float(_rd_cfg.get("min_gap",          5.0)),
            "max_gap":         float(_rd_cfg.get("max_gap",         30.0)),
            "flat_start":      float(_rd_cfg.get("flat_start",      10.0)),
        }
        if road_override_kwargs:
            self._random_road_kwargs.update(road_override_kwargs)
        self._v_random_low = self._rcfg.v_min * float(_rd_cfg.get("v_random_low_factor", 2.0))

        # filter state — cleared in reset()
        self._prev_a         = 0.0
        self._filtered_a     = 0.0
        self._filtered_jerk  = 0.0
        self._prev_action    = 0.0
        self._last_preview_max = 0.0

        self._fig            = None
        self._ren_hist       = None
        self._episode_count  = 0
        self._show_ts        = bool(show_time_series)
        self._freeze_episode = bool(freeze_episode)
        self._freeze_render  = False
        self._ts_flags = {
            "z": bool(show_ts_z),
            "speed": bool(show_ts_speed),
            "z_ddot": bool(show_ts_z_ddot),
        }

    # gymnasium interface

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self._rcfg
        opts = options or {}

        randomize       = opts.get("randomize_road",  self._random_road_on_reset)
        randomize_speed = opts.get("randomize_speed", True)

        if self.road_profile == 'speed_bump' and "road" in opts:
            # caller supplies a pre-built RoadGenerator (e.g. eval scenarios)
            self._road = opts["road"]
            self._v    = self._road.speed
        elif self.road_profile == 'speed_bump' and randomize:
            road_kwargs = opts.get("road_kwargs", self._random_road_kwargs)
            v_low  = float(opts.get("v_random_low",  self._v_random_low))
            v_high = float(opts.get("v_random_high", cfg.v_max))
            v = float(rng.uniform(v_low, v_high)) if randomize_speed else self._v0
            self._road = RoadGenerator.from_random(
                rng, vehicle_speed=v, **road_kwargs
            )
            self._v = self._road.speed
        else:
            self._road.reset(seed=int(rng.integers(0, 2**31)))
            self._road.set_speed(self._v0)
            self._v = self._v0

        v_init = self._v   # set by road branch above; use it to seed ODE speed slot
        x = self._ode.reset(v_init)
        if not self._start_at_eq:
            x[0:4] += rng.normal(0.0, 0.005, size=4)
            x[5]   += rng.normal(0.0, 0.001)
        self._state = x

        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._last_z_B_ddot  = 0.0
        self._last_z_W_ddot  = 0.0
        self._episode_reward = 0.0
        self._episode_count += 1

        if self._fig is not None and self.render_mode == 'human':
            # keep the window open between episodes — just clear the history
            if self._ren_hist is not None:
                for buf in self._ren_hist.values():
                    buf.clear()
        else:
            if self._fig is not None:
                close_env(self)
            self._ren_hist = None

        self._v_ref_last     = self._rcfg.v_max
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        # recompute after road regeneration — random roads may have different length
        self._max_distance = self._compute_max_distance()

        # bump-crossing state — reset each episode
        self._bumps_passed   = 0
        self._bump_ends      = sorted(
            x0 + L + 1.0 for (x0, _, L) in self._road._bumps
        ) if self.road_profile == 'speed_bump' else []

        # reset filter state
        self._prev_a         = 0.0
        self._filtered_a     = 0.0
        self._filtered_jerk  = 0.0
        self._prev_action    = 0.0
        self._last_preview_max = 0.0

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
        self._road.set_speed(v_new)          # also geometry-clamps road.speed
        s_pos_start    = self._s_pos         # position at START of this step
        self._s_pos   += v_new * DT

        # 2. Update IIR filters
        alpha_a = cfg.accel_filter_alpha
        a_clipped = float(np.clip(a_actual, -cfg.accel_clip, cfg.accel_clip))
        self._filtered_a = alpha_a * self._filtered_a + (1.0 - alpha_a) * a_clipped

        jerk = (a_actual - self._prev_a) / DT
        alpha_j = cfg.jerk_filter_alpha
        j_clipped = float(np.clip(jerk, -cfg.jerk_clip, cfg.jerk_clip))
        self._filtered_jerk = alpha_j * self._filtered_jerk + (1.0 - alpha_j) * j_clipped

        # 3. Integrate ODE one control step (position-based — no drift)
        v_road = self._road.speed   # post-geometry-clamp value
        new_state, z_B_ddot, z_W_ddot = self._ode.step(
            self._state, self._road, s_pos_start, v_road
        )
        self._state         = new_state
        self._t            += DT
        self._step_count   += 1
        self._last_z_B_ddot = z_B_ddot
        self._last_z_W_ddot = z_W_ddot

        travel = float(new_state[2])   # used for truncation check below

        self._accel_sq  += z_B_ddot ** 2
        self._peak_accel = max(self._peak_accel, abs(z_B_ddot))

        # 4. Speed band upper limit and reward
        v_ref    = self._compute_v_ref(self._t)
        self._v_ref_last = v_ref

        reward, breakdown = compute_reward(
            v_new,
            self._last_z_B_ddot,
            self._last_z_W_ddot,
            self._filtered_a,
            self._filtered_jerk,
            self._prev_action, u,
            cfg,
        )
        self._speed_err_sq   += (v_ref - v_new) ** 2

        # 4b. Bump-crossing reward — fire once per bump when s_pos clears its end
        r_bumps = 0.0
        while (self._bumps_passed < len(self._bump_ends)
               and self._s_pos >= self._bump_ends[self._bumps_passed]):
            self._bumps_passed += 1
            r_bumps += cfg.w_bump_cross
        reward               += r_bumps
        breakdown["r_bumps"]  = r_bumps

        self._episode_reward += reward

        # 5. Update history
        self._prev_a      = a_actual
        self._prev_action = u

        # 6. Termination / truncation
        #
        # Physical safety truncation: numerical blow-up or extreme state.
        # Thresholds set above the worst observed values (0.094 m travel, 0.28 m z_B)
        # with margin — fire only if something has gone badly wrong.
        truncated = bool(
            abs(travel) > self._trunc_travel          # suspension blow-up
            or abs(float(new_state[5])) > self._trunc_zs  # body blow-up
        )
        terminated = False

        # Normal termination: step budget exhausted OR road fully cleared.
        # Both cases give a terminal bonus so the agent is rewarded for quality,
        # not just for lasting the full 300 steps.
        road_complete = (
            self._max_distance is not None
            and self._s_pos >= self._max_distance
        )
        if (self._step_count >= self._max_episode_steps or road_complete) and not truncated:
            terminated = True
            rms        = np.sqrt(self._accel_sq / self._step_count)
            mean_speed = self._s_pos / max(self._t, 1e-9)
            reward    += compute_terminal_bonus(rms, mean_speed, cfg)

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

    # observation and info helpers

    def _build_obs_bounds(self):
        cfg     = self._rcfg
        a_bound = cfg.accel_clip / cfg.a_comfort
        j_bound = cfg.jerk_clip  / cfg.j_max

        # base obs: [ζ, ζ̇, v/v_max, filtered_a, filtered_jerk, prev_action]
        # PreviewWrapper appends the peak slots on top of this
        extra_high = np.array([1.0, a_bound, j_bound, 1.0], dtype=np.float32)
        extra_low  = np.array([0.0, -a_bound, -j_bound, -1.0], dtype=np.float32)
        high = np.concatenate([OBS_HIGH, extra_high])
        low  = np.concatenate([OBS_LOW,  extra_low])
        return high, low

    def _obs(self) -> np.ndarray:
        cfg = self._rcfg

        # use actual arc-length position, not speed × time (Bug-fix: position drift)
        zeta     = self._road.get_height_at(self._s_pos)
        zeta_dot = self._road.get_height_dot_at(self._s_pos, self._v)
        base_obs = np.clip(
            np.array([zeta, zeta_dot], dtype=np.float32), OBS_LOW, OBS_HIGH
        )

        a_bound = cfg.accel_clip / cfg.a_comfort
        j_bound = cfg.jerk_clip  / cfg.j_max
        scalars = np.array([
            float(np.clip(self._v / cfg.v_max, 0.0, 1.0)),
            float(np.clip(self._filtered_a / cfg.a_comfort, -a_bound, a_bound)),
            float(np.clip(self._filtered_jerk / cfg.j_max, -j_bound, j_bound)),
            float(self._prev_action),
        ], dtype=np.float32)

        return np.concatenate([base_obs, scalars])

    def _info(self, z_B_ddot: float) -> dict:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return {
            'episode_reward': float(self._episode_reward),
            'rms_accel':       float(rms),
            'peak_accel':      float(self._peak_accel),
            'comfort_score':   float(max(0.0, 1.0 - rms / self._rcfg.a_limit)),
            'road_profile':    self.road_profile,
            'step_count':      self._step_count,
            'episode_time':    self._t,
            'z_B_ddot':        float(z_B_ddot),
            'z_W_ddot':        float(self._last_z_W_ddot),
            'speed':           float(self._v),              # m/s  (internal)
            'speed_kmh':       float(self._v * 3.6),        # km/h (display)
            'v_ref':           float(self._v_ref_last),
            'v_ref_kmh':       float(self._v_ref_last * 3.6),
            'speed_error':       float(self._v_ref_last - self._v),
            'speed_error_rms':   float(np.sqrt(self._speed_err_sq / n)),
            'bumps_passed':    int(self._bumps_passed),
            'bumps_total':     len(self._bump_ends),
        }

    def get_comfort_metric(self) -> float:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return float(max(0.0, 1.0 - rms / self._rcfg.a_limit))

    # speed reference profile

    def _compute_max_distance(self) -> Optional[float]:
        # work out a sensible max_distance from the road layout and step budget:
        #   speed_bump: last bump end + 5 m, capped by v_max * episode budget
        #   recorded: full track length
        #   flat/iso/etc: None (step count terminates instead)
        episode_budget_m = self._max_episode_steps * DT * self._rcfg.v_max

        if self.road_profile == 'speed_bump' and self._road._bumps:
            x0, _, L = self._road._bumps[-1]
            last_bump_end = x0 + L
            return min(last_bump_end + 5.0, episode_budget_m)

        if self.road_profile == 'recorded' and self._road._rec_arc is not None:
            return min(float(self._road._rec_arc[-1]), episode_budget_m)

        return None  # flat: no distance limit

    def _compute_v_ref(self, t: float) -> float:
        if self._ref_speed_profile == "custom" and self._v_ref_fn is not None:
            return float(self._v_ref_fn(t))
        cfg = self._rcfg
        if self.road_profile != 'speed_bump' or not self._road._bumps:
            return cfg.v_max

        s = self._s_pos
        best_d: float | None = None
        best_h: float = 0.0
        for x0, A, L in self._road._bumps:
            if x0 + L <= s:
                continue                      # already passed
            d_to_entry = max(0.0, x0 - s)    # 0 when car is on the bump
            if d_to_entry > cfg.preview_distance:
                continue                      # beyond horizon
            if best_d is None or d_to_entry < best_d:
                best_d = d_to_entry
                best_h = A                    # peak height = A (cosine profile)

        if best_d is None or best_h < cfg.peak_height_min:
            return cfg.v_max

        h_ratio   = float(min(1.0, best_h / cfg.h_clip))
        proximity = float(max(0.0, 1.0 - best_d / cfg.preview_distance))
        # up to 50% reduction at the bump face; tapers with distance
        v_ref = cfg.v_max * (1.0 - 0.5 * h_ratio * proximity)
        return float(max(cfg.v_min, v_ref))
    



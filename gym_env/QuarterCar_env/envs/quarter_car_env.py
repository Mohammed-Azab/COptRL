import os
import collections
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from QuarterCar_env.ode_model import QuarterCarODE
from QuarterCar_env.road_generator import RoadGenerator
from QuarterCar_env.reward import RewardConfig, compute_reward, compute_terminal_bonus
from QuarterCar_env.params import (
    F_MAX, DT, EPISODE_STEPS,
    TRUNC_TRAVEL, TRUNC_ZS, MAX_DISTANCE,
    OBS_HIGH, OBS_LOW,
    VEHICLE_SPEED,
    V_TAU, V_MAX, V_MIN, V_BRAKE_LEAD,
)
from QuarterCar_env.render_params import (
    RENDER_Y_SCALE, RENDER_HIST_SECS,
    RENDER_SHOW_TIMESERIES, RENDER_N_TIMESERIES,
    RENDER_Y_W_NOM, RENDER_Y_B_NOM,
    RENDER_H_MW, RENDER_W_MW,
    RENDER_H_MB, RENDER_W_MB,
    RENDER_XLIM, RENDER_YLIM,
    RENDER_ROAD_HALF, RENDER_ROAD_N,
    RENDER_C_MB, RENDER_C_MW, RENDER_C_SPRING, RENDER_C_DAMPER,
    RENDER_C_ROAD, RENDER_C_GROUND,
    RENDER_SP_X, RENDER_SP_W, RENDER_SP_N,
    RENDER_DA_X, RENDER_DA_W, RENDER_DA_PIST_H, RENDER_DA_PIST_FRAC,
    RENDER_DA_LOWER_STEM, RENDER_DA_CYL_H_SUSP, RENDER_DA_CYL_H_TIRE,
    RENDER_CONTACT_STEM, RENDER_GROUND_Y,
    Y_LINE_OFFSET,
)

_ROAD_X = np.linspace(-RENDER_ROAD_HALF, RENDER_ROAD_HALF, RENDER_ROAD_N)
_N_HIST = int(RENDER_HIST_SECS / DT)


#  Render geometry helpers 

def _spring_xy(x_c: float, y_top: float, y_bot: float,
               n: int = 8, w: float = 0.18):
    """Zigzag coil spring; returns (xs, ys) for a single Line2D."""
    n_pts = 2 * n + 2
    ys = np.linspace(y_top, y_bot, n_pts)
    xs = np.full(n_pts, x_c)
    idx = np.arange(1, n_pts - 1)
    xs[1:-1] = x_c + w * np.where(idx % 2 == 1, 1.0, -1.0)
    return xs, ys


def _damper_xy(x_c: float, y_top: float, y_bot: float, cyl_h: float):
    """
    Open-top piston-cylinder damper (⊔).
    y_bot: lower mass attachment; y_top: upper mass attachment.
    A short lower rod links y_bot → cylinder base (RENDER_DA_LOWER_STEM).
    The upper rod links piston top → y_top.
    Returns (upper_rod_xy, lower_rod_xy, cyl_xy, pist_rect).
    """
    hw      = RENDER_DA_W / 2
    gap     = max(y_top - y_bot, 0.05)
    cyl_bot = y_bot + RENDER_DA_LOWER_STEM       # cylinder sits above lower mass
    cyl_top = cyl_bot + cyl_h
    pist_h  = RENDER_DA_PIST_H
    pist_top = float(np.clip(
        y_bot + gap * RENDER_DA_PIST_FRAC,
        cyl_bot + pist_h + 0.005,
        cyl_top - 0.005,
    ))
    pist_bot = pist_top - pist_h
    upper_rod_xy = ([x_c, x_c], [pist_top, y_top])          # piston top → upper mass
    lower_rod_xy = ([x_c, x_c], [y_bot,    cyl_bot])         # lower mass → cylinder base
    cyl_xy       = ([x_c - hw, x_c - hw, x_c + hw, x_c + hw],
                    [cyl_top,  cyl_bot,   cyl_bot,   cyl_top])
    m = 0.01
    pist_rect = (x_c - hw + m, pist_bot, 2 * hw - 2 * m, pist_h)
    return upper_rod_xy, lower_rod_xy, cyl_xy, pist_rect


def _ground_symbol_xy(x_c: float, y: float, half_w: float = 0.55):
    """Horizontal ground line only."""
    return np.array([x_c - half_w, x_c + half_w]), np.array([y, y])


# Environment

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
        render_show_timeseries: bool = RENDER_SHOW_TIMESERIES,
        render_n_timeseries: int = RENDER_N_TIMESERIES,
        control_mode: str = "suspension",       # "suspension" | "speed" | "hybrid"
        v_max: float = V_MAX,                   # m/s — maximum longitudinal speed
        ref_speed_profile: str = "constant",    # "constant" | "slow_before_bump" | "custom"
        max_episode_steps: int = EPISODE_STEPS, # steps before terminated=True
        max_distance: Optional[float] = MAX_DISTANCE,  # m — truncate when s_pos exceeds this
    ):
        super().__init__()
        self.render_mode  = render_mode
        self.road_profile = road_profile
        self._v0          = float(vehicle_speed)
        self._y_scale     = int(render_y_scale)
        self._show_ts     = bool(render_show_timeseries)
        self._n_ts        = max(1, min(4, int(render_n_timeseries)))

        if control_mode not in ("suspension", "speed", "hybrid"):
            raise ValueError(f"control_mode must be 'suspension', 'speed', or 'hybrid', got {control_mode!r}")
        self._control_mode      = control_mode
        self._v_max             = float(v_max)
        self._v_min             = V_MIN
        self._ref_speed_profile = ref_speed_profile
        self._max_episode_steps = int(max_episode_steps)
        self._max_distance      = max_distance

        # Optional custom v_ref callable passed through road_params
        self._v_ref_fn: Optional[Callable[[float], float]] = (road_params or {}).get('v_ref_fn', None)

        # action space
        if control_mode == "suspension":
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([ 1.0], dtype=np.float32),
            )
        elif control_mode == "speed":
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
            )
        else:  # "hybrid"
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([ 1.0, 1.0], dtype=np.float32),
            )

        # observation space
        if control_mode == "suspension":
            self.observation_space = spaces.Box(
                low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        else:
            obs_high_ext = np.concatenate(
                [OBS_HIGH, [self._v_max, self._v_max]]).astype(np.float32)
            obs_low_ext  = np.concatenate(
                [OBS_LOW,  [0.0,        -self._v_max]]).astype(np.float32)
            self.observation_space = spaces.Box(
                low=obs_low_ext, high=obs_high_ext, dtype=np.float32)

        self._ode  = QuarterCarODE(physics_params)
        self._road = RoadGenerator(road_profile, vehicle_speed, road_params)
        self._rcfg = reward_config or RewardConfig()

        # episode state
        self._state          = self._ode.reset(self._v0)
        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._travel_sq      = 0.0
        self._last_F         = 0.0
        self._last_z_B_ddot  = 0.0
        self._episode_reward = 0.0

        # speed control state
        self._v              = self._v0
        self._v_ref_last     = self._v_max
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0   # accumulated longitudinal distance (m)

        # precompute bump times used by slow_before_bump profile
        self._bump_times: list = self._road.get_bump_times()

        self._fig            = None
        self._ren_hist       = None
        self._fd_arrow_patch = None
        self._episode_count  = 0

    # Gymnasium interface

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self._road.reset(seed=int(rng.integers(0, 2**31)))
        self._road.set_speed(self._v0)

        x = self._ode.reset(self._v0)
        x[0:4] += rng.normal(0.0, 0.005, size=4)
        x[5]   += rng.normal(0.0, 0.001)
        self._state = x

        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._travel_sq      = 0.0
        self._last_F         = 0.0
        self._last_z_B_ddot  = 0.0
        self._episode_reward = 0.0
        self._episode_count += 1

        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
        self._ren_hist       = None

        self._v              = self._v0
        self._v_ref_last     = self._v_max
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        # rebuild bump times after road reset (bump_x_start may differ across resets)
        self._bump_times = self._road.get_bump_times()

        return self._obs(), self._info(0.0)

    def step(self, action):
        # 1. Parse action, update speed if applicable
        if self._control_mode == "suspension":
            F_act = float(np.clip(action[0], -1.0, 1.0)) * F_MAX
        elif self._control_mode == "speed":
            F_act = 0.0
            v_cmd = float(np.clip(action[0], 0.0, 1.0)) * self._v_max
            v_new = self._v + DT * (v_cmd - self._v) / V_TAU
            self._v = float(np.clip(v_new, 0.0, self._v_max))
            self._state[4] = self._v
            self._road.set_speed(self._v)
        else:  # "hybrid"
            F_act = float(np.clip(action[0], -1.0, 1.0)) * F_MAX
            v_cmd = float(np.clip(action[1], 0.0, 1.0)) * self._v_max
            v_new = self._v + DT * (v_cmd - self._v) / V_TAU
            self._v = float(np.clip(v_new, 0.0, self._v_max))
            self._state[4] = self._v
            self._road.set_speed(self._v)

        self._last_F = F_act
        self._s_pos += self._v * DT

        # 2. Integrate ODE
        new_state, z_B_ddot = self._ode.step(
            self._state, self._road.get_height_dot, self._t, F_act
        )
        self._state         = new_state
        self._t            += DT
        self._step_count   += 1
        self._last_z_B_ddot = z_B_ddot

        tyre_defl = float(new_state[0])
        travel    = float(new_state[2])

        self._accel_sq   += z_B_ddot ** 2
        self._travel_sq  += travel ** 2
        self._peak_accel  = max(self._peak_accel, abs(z_B_ddot))

        # 3. Reference speed and reward
        v_ref = self._compute_v_ref(self._t)
        self._v_ref_last = v_ref

        if self._control_mode == "suspension":
            reward = compute_reward(z_B_ddot, travel, tyre_defl, F_act, self._rcfg)
        else:
            self._rcfg.v_ref = v_ref
            reward = compute_reward(
                z_B_ddot, travel, tyre_defl, F_act, self._rcfg, v=self._v)
            speed_err = v_ref - self._v
            self._speed_err_sq += speed_err ** 2

        self._episode_reward += reward

        # 4. Termination
        truncated  = bool(
            abs(travel) > TRUNC_TRAVEL
            or abs(float(new_state[5])) > TRUNC_ZS
            or (self._max_distance is not None and self._s_pos >= self._max_distance)
        )
        terminated = False

        if self._step_count >= self._max_episode_steps and not truncated:
            terminated = True
            rms = np.sqrt(self._accel_sq / self._step_count)
            reward += compute_terminal_bonus(rms, self._rcfg)

        if self.render_mode == 'human':
            self.render()

        return self._obs(), reward, terminated, truncated, self._info(z_B_ddot)

    def render(self):
        if self.render_mode == 'none':
            return None
        if self._ren_hist is None:
            self._init_render()
        self._push_history()
        self._update_artists()
        if self.render_mode == 'human':
            import matplotlib.pyplot as plt
            self._fig.canvas.draw_idle()
            plt.pause(1e-3)
            return None
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        w, h = self._fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return img[..., :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    # Observation / info

    def _obs(self) -> np.ndarray:
        x        = self._state
        zeta     = self._road.get_height(self._t)
        zeta_dot = self._road.get_height_dot(self._t)
        z_B = float(x[5])
        z_W = z_B + float(x[2])
        raw = np.array([
            z_B,          # 0: body displacement
            float(x[3]),  # 1: ż_B
            z_W,          # 2: wheel displacement
            float(x[1]),  # 3: ż_W
            zeta,         # 4: road height
            zeta_dot,     # 5: zeta_dot
            float(x[2]),  # 6: suspension travel
            float(x[0]),  # 7: tyre deflection
        ], dtype=np.float32)
        base_obs = np.clip(raw, OBS_LOW, OBS_HIGH)

        if self._control_mode == "suspension":
            return base_obs

        v_ref = self._v_ref_last
        v     = self._v
        speed_ext = np.array([
            np.clip(v,         0.0,          self._v_max),   # 8: current speed [m/s]
            np.clip(v_ref - v, -self._v_max, self._v_max),   # 9: speed error v_ref − v [m/s]
        ], dtype=np.float32)
        return np.concatenate([base_obs, speed_ext])

    def _info(self, z_B_ddot: float) -> dict:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        info = {
            'rms_accel':      float(rms),
            'peak_accel':     float(self._peak_accel),
            'suspension_rms': float(np.sqrt(self._travel_sq / n)),
            'comfort_score':  float(max(0.0, 1.0 - rms / self._rcfg.a_limit)),
            'road_profile':   self.road_profile,
            'step_count':     self._step_count,
            'episode_time':   self._t,
            'F_act':          self._last_F,
            'z_B_ddot':       float(z_B_ddot),
        }
        if self._control_mode != "suspension":
            n_s = max(self._step_count, 1)
            info.update({
                'speed':            float(self._v),
                'v_ref':            float(self._v_ref_last),
                'speed_error':      float(self._v_ref_last - self._v),
                'speed_error_rms':  float(np.sqrt(self._speed_err_sq / n_s)),
            })
        return info

    def get_comfort_metric(self) -> float:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return float(max(0.0, 1.0 - rms / self._rcfg.a_limit))

    # Speed reference profile

    def _compute_v_ref(self, t: float) -> float:
        if self._ref_speed_profile == "constant":
            return self._v_max
        if self._ref_speed_profile == "custom":
            return float(self._v_ref_fn(t))
        if self._ref_speed_profile == "slow_before_bump":
            times = self._bump_times
            if not times:
                return self._v_max
            t_start, t_center, t_end = times
            t_brake_start = t_center - V_BRAKE_LEAD
            t_accel_end   = t_end    + V_BRAKE_LEAD
            if t < t_brake_start:
                return self._v_max
            if t < t_center:
                alpha = (t - t_brake_start) / V_BRAKE_LEAD
                return self._v_max - (self._v_max - self._v_min) * alpha
            if t <= t_end:
                return self._v_min
            if t <= t_accel_end:
                alpha = (t - t_end) / V_BRAKE_LEAD
                return self._v_min + (self._v_max - self._v_min) * alpha
            return self._v_max
        return self._v_max

    #  Render internals

    def _init_render(self):
        """Build the figure and all artists exactly once."""
        import matplotlib
        if not os.environ.get('DISPLAY'):
            matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle

        self._ren_hist = {
            't':        collections.deque(maxlen=_N_HIST),
            'z_B':      collections.deque(maxlen=_N_HIST),
            'z_W':      collections.deque(maxlen=_N_HIST),
            'z_B_ddot': collections.deque(maxlen=_N_HIST),
            'F':        collections.deque(maxlen=_N_HIST),
            's_dot':    collections.deque(maxlen=_N_HIST),
        }

        # figure layout
        win_title = f'Quarter_Car Model : ep{self._episode_count}'
        if self._show_ts:
            fig = plt.figure(figsize=(14, 7))
            gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 2],
                           left=0.06, right=0.97, bottom=0.09, top=0.93, wspace=0.38)
            ax_s = fig.add_subplot(gs[0, 0])
            gs_r = gs[0, 1].subgridspec(self._n_ts, 1, hspace=0.10)
            ax_r = [fig.add_subplot(gs_r[i]) for i in range(self._n_ts)]
        else:
            fig  = plt.figure(figsize=(9, 7))
            ax_s = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.93)
            ax_r = []

        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(win_title)

        # schematic axis
        ax_s.set_facecolor('white')
        ax_s.set_xlim(RENDER_XLIM)
        ax_s.set_ylim(RENDER_YLIM)
        ax_s.set_xlabel('position relative to car (m)', fontsize=9)
        ax_s.set_ylabel(f'height  (m × {self._y_scale})', fontsize=9)
        ax_s.tick_params(labelsize=8)
        ax_s.spines[['top', 'right']].set_visible(False)

        # road profile — gray line, no fill
        road_line, = ax_s.plot([], [], '-', color=RENDER_C_ROAD, lw=1.5, zorder=2,
                               label='road profile ζ(x)')

        # ground symbol — updated each frame to track road surface under the car
        ground_sym, = ax_s.plot([], [], '-', color=RENDER_C_GROUND, lw=1.5, zorder=2)

        # contact stem + dot (stem goes below ground line, dot at bottom)
        contact_stem, = ax_s.plot([], [], '-', color=RENDER_C_GROUND, lw=1.8, zorder=6)
        contact_dot,  = ax_s.plot([], [], 'o', color=RENDER_C_GROUND, ms=10, zorder=7)

        # tire elements (k_T left, c_T right)
        _hw = RENDER_DA_W / 2
        tire_spring,          = ax_s.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0, zorder=4)
        tire_damp_rod,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        tire_damp_lower_rod,  = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        tire_damp_cyl,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0, zorder=4)
        tire_damp_pist = Rectangle(
            (RENDER_DA_X - _hw + 0.01, 0), RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
            fc=RENDER_C_DAMPER, ec='none', zorder=4)
        ax_s.add_patch(tire_damp_pist)

        # m_W block — steel blue, dot inside
        mw_patch = Rectangle(
            (-RENDER_W_MW / 2, RENDER_Y_W_NOM - RENDER_H_MW / 2),
            RENDER_W_MW, RENDER_H_MW,
            fc=RENDER_C_MW, ec='black', lw=1.5, zorder=5,
        )
        ax_s.add_patch(mw_patch)
        mw_dot   = ax_s.plot(0, RENDER_Y_W_NOM, 'o', color='black', ms=5, zorder=7)[0]
        mw_label = ax_s.text(-RENDER_W_MW / 2 + 0.05, RENDER_Y_W_NOM,
                              r'$m_W$', ha='left', va='center',
                              fontsize=9, fontweight='bold', color='white', zorder=7)

        # suspension elements (k_S left, c_S right)
        susp_spring,          = ax_s.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0, zorder=4)
        susp_damp_rod,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        susp_damp_lower_rod,  = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        susp_damp_cyl,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0, zorder=4)
        susp_damp_pist = Rectangle(
            (RENDER_DA_X - _hw + 0.01, 0), RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
            fc=RENDER_C_DAMPER, ec='none', zorder=4)
        ax_s.add_patch(susp_damp_pist)

        # m_B block — golden yellow, dot inside
        mb_patch = Rectangle(
            (-RENDER_W_MB / 2, RENDER_Y_B_NOM - RENDER_H_MB / 2),
            RENDER_W_MB, RENDER_H_MB,
            fc=RENDER_C_MB, ec='black', lw=1.5, zorder=5,
        )
        ax_s.add_patch(mb_patch)
        mb_dot   = ax_s.plot(0, RENDER_Y_B_NOM, 'o', color='black', ms=5, zorder=7)[0]
        mb_label = ax_s.text(-RENDER_W_MB / 2 + 0.05, RENDER_Y_B_NOM,
                              r'$m_B$', ha='left', va='center',
                              fontsize=9, fontweight='bold', color='black', zorder=7)

        # F_D arrow — remove/add each frame (only allocation in hot-path)
        fd_text = ax_s.text(0, RENDER_Y_B_NOM, '',
                            ha='left', va='center', fontsize=8,
                            fontweight='bold', color='#0055cc', zorder=8)

        # status text — top-left corner
        status_text = ax_s.text(
            0.02, 0.98, '', transform=ax_s.transAxes,
            va='top', ha='left', fontsize=7.5, family='monospace',
            bbox=dict(facecolor='white', alpha=0.80, edgecolor='#cccccc',
                      boxstyle='round,pad=0.3'),
            zorder=9,
        )

        # exaggeration note
        ax_s.text(0.98, 0.02, f'y ×{self._y_scale}',
                  transform=ax_s.transAxes,
                  va='bottom', ha='right', fontsize=7, color='#aaaaaa', zorder=9)

        ax_s.legend(fontsize=7, loc='upper right', framealpha=0.7,
                    handlelength=1.5, borderpad=0.4)

        # time-series axes
        _ts_specs = [
            # (key_B,   key_W,       ylabel,                  color_B, color_W)
            ('z_B',    'z_W',       'z (m)',                 'b',     'r'),
            ('z_B_ddot', None,      r'$\ddot{z}_B$ (m/s²)',  'k',     None),
            ('F',        None,      r'$F_D$ (N)',            '#008800', None),
            ('s_dot',    None,      r'$\dot{s}$ (m/s)',      '#aa00aa', None),
        ]
        ts = {}
        for i, ax in enumerate(ax_r):
            k1, k2, ylabel, c1, c2 = _ts_specs[i]
            ts[k1], = ax.plot([], [], '-', color=c1, lw=1,
                              label=(r'$z_B$' if k1 == 'z_B' else None))
            if k2:
                ts[k2], = ax.plot([], [], '--', color=c2, lw=1, label=r'$z_W$')
            if k1 == 'z_B':
                ax.legend(fontsize=7, loc='upper left', framealpha=0.6)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.axhline(0, color='gray', lw=0.4)
            if i < len(ax_r) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('t (s)', fontsize=8)

        if self.render_mode == 'human':
            plt.ion()
            plt.show(block=False)

        self._fig  = fig
        self._ax_s = ax_s
        self._ax_r = ax_r
        self._artists = {
            'road_line':      road_line,
            'contact_stem':   contact_stem,
            'contact_dot':    contact_dot,
            'ground_sym':     ground_sym,
            'tire_spring':         tire_spring,
            'tire_damp_rod':       tire_damp_rod,
            'tire_damp_lower_rod': tire_damp_lower_rod,
            'tire_damp_cyl':       tire_damp_cyl,
            'tire_damp_pist':      tire_damp_pist,
            'mw_patch':            mw_patch,
            'mw_dot':              mw_dot,
            'mw_label':            mw_label,
            'susp_spring':         susp_spring,
            'susp_damp_rod':       susp_damp_rod,
            'susp_damp_lower_rod': susp_damp_lower_rod,
            'susp_damp_cyl':       susp_damp_cyl,
            'susp_damp_pist':      susp_damp_pist,
            'mb_patch':       mb_patch,
            'mb_dot':         mb_dot,
            'mb_label':       mb_label,
            'fd_text':        fd_text,
            'status_text':    status_text,
            'ts':             ts,
        }

    def _push_history(self):
        x   = self._state
        z_B = float(x[5])
        z_W = z_B + float(x[2])
        h   = self._ren_hist
        h['t'].append(self._t)
        h['z_B'].append(z_B)
        h['z_W'].append(z_W)
        h['z_B_ddot'].append(self._last_z_B_ddot)
        h['F'].append(self._last_F)
        h['s_dot'].append(float(self._v))

    def _update_artists(self):
        """Update all artists in-place. Only the F_D arrow patch is re-allocated."""
        from matplotlib.patches import FancyArrow

        art = self._artists
        ys  = self._y_scale
        x   = self._state

        z_B    = float(x[5])
        z_W    = z_B + float(x[2])
        F_act  = self._last_F
        zeta_0 = float(self._road.get_height(self._t))

        # draw-space heights for the two masses (RENDER_GROUND_Y shifts entire system)
        y_W      = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_W * ys
        y_B      = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_B * ys
        y_road_0 = RENDER_GROUND_Y + zeta_0 * ys   # road surface directly below car

        # road profile (gray line, car at x=0, road scrolls left)
        v_disp  = max(self._v, 0.1)
        t_q     = self._t + _ROAD_X / v_disp
        road_h  = self._road.get_height_array(t_q) * ys + RENDER_GROUND_Y
        art['road_line'].set_data(_ROAD_X, road_h)

        h = self._ren_hist

        y_road_0 += Y_LINE_OFFSET
        # ground symbol + contact stem + dot
        gx, gy = _ground_symbol_xy(0.0, y_road_0, half_w=RENDER_W_MB / 2 + 0.15)
        art['ground_sym'].set_data(gx, gy)
        art['contact_stem'].set_data([0.0, 0.0], [y_road_0 - Y_LINE_OFFSET + RENDER_CONTACT_STEM, y_road_0])
        art['contact_dot'].set_data([0.0], [y_road_0-Y_LINE_OFFSET])

        # tire spring (k_T) and tire damper (c_T)
        y_tire_top = y_W - RENDER_H_MW / 2
        art['tire_spring'].set_data(
            *_spring_xy(RENDER_SP_X, y_tire_top, y_road_0, RENDER_SP_N, RENDER_SP_W))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(RENDER_DA_X, y_tire_top, y_road_0,
                                              RENDER_DA_CYL_H_TIRE)
        art['tire_damp_rod'].set_data(*u_rod)
        art['tire_damp_lower_rod'].set_data(*l_rod)
        art['tire_damp_cyl'].set_data(*cyl_xy)
        art['tire_damp_pist'].set_xy((pr[0], pr[1]))
        art['tire_damp_pist'].set_height(pr[3])

        # m_W block
        art['mw_patch'].set_xy((-RENDER_W_MW / 2, y_W - RENDER_H_MW / 2))
        art['mw_dot'].set_data([0], [y_W])
        art['mw_label'].set_position((-RENDER_W_MW / 2 + 0.05, y_W))

        # suspension spring (k_S) and damper (c_S)
        y_susp_bot = y_W + RENDER_H_MW / 2
        y_susp_top = y_B - RENDER_H_MB / 2
        if y_susp_top > y_susp_bot + 0.05:
            art['susp_spring'].set_data(
                *_spring_xy(RENDER_SP_X, y_susp_top, y_susp_bot, RENDER_SP_N, RENDER_SP_W))
            u_rod, l_rod, cyl_xy, pr = _damper_xy(RENDER_DA_X, y_susp_top, y_susp_bot,
                                                  RENDER_DA_CYL_H_SUSP)
            art['susp_damp_rod'].set_data(*u_rod)
            art['susp_damp_lower_rod'].set_data(*l_rod)
            art['susp_damp_cyl'].set_data(*cyl_xy)
            art['susp_damp_pist'].set_xy((pr[0], pr[1]))
            art['susp_damp_pist'].set_height(pr[3])

        # m_B block
        art['mb_patch'].set_xy((-RENDER_W_MB / 2, y_B - RENDER_H_MB / 2))
        art['mb_dot'].set_data([0], [y_B])
        art['mb_label'].set_position((-RENDER_W_MB / 2 + 0.05, y_B))

        #  F_D arrow (remove old patch, add fresh one)
        if self._fd_arrow_patch is not None:
            self._fd_arrow_patch.remove()
            self._fd_arrow_patch = None
        arrow_len = (F_act / F_MAX) * 1.0   # max 1.0 draw unit
        if abs(arrow_len) > 0.04:
            x_start = RENDER_W_MB / 2 if F_act > 0 else -RENDER_W_MB / 2
            al      = abs(arrow_len)
            arr = FancyArrow(x_start, y_B, arrow_len, 0.0,
                             width=0.035, length_includes_head=True,
                             head_width=0.12, head_length=min(al * 0.25, 0.22),
                             fc='#0055cc', ec='#0055cc', zorder=8)
            self._ax_s.add_patch(arr)
            self._fd_arrow_patch = arr
            txt_x = x_start + arrow_len + (0.08 if F_act > 0 else -0.08)
            art['fd_text'].set_text(f'$F_D$={F_act:+.0f} N')
            art['fd_text'].set_position((txt_x, y_B))
            art['fd_text'].set_ha('left' if F_act > 0 else 'right')
        else:
            art['fd_text'].set_text('')

        #  status text 
        art['status_text'].set_text(
            f't={self._t:6.2f} s    s={self._s_pos:6.1f} m\n'
            f'z_B={z_B*100:+.2f} cm  z_W={z_W*100:+.2f} cm\n'
            f'ζ={zeta_0*100:.3f} cm    '
            f'F_D={F_act:+.0f} N\n'
            f'v={self._v:.1f} m/s    '
            f'ep reward={self._episode_reward:.2f}'
        )

        #  time-series 
        if not self._show_ts:
            return
        t_arr = np.array(h['t'])
        ts    = art['ts']
        _map = {
            'z_B':      h['z_B'],
            'z_W':      h['z_W'],
            'z_B_ddot': h['z_B_ddot'],
            'F':        h['F'],
            's_dot':    h['s_dot'],
        }
        for key, line in ts.items():
            line.set_data(t_arr, np.array(_map[key]))

        if len(t_arr) > 1:
            t_hi = t_arr[-1]
            for ax in self._ax_r:
                ax.set_xlim(t_hi - RENDER_HIST_SECS, t_hi)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

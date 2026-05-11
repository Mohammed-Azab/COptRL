import os
import collections

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
    RENDER_Y_SCALE, RENDER_HIST_SECS,
    RENDER_Y_W_NOM, RENDER_Y_B_NOM,
    RENDER_H_MW, RENDER_W_MW,
    RENDER_H_MB, RENDER_W_MB,
    RENDER_XLIM, RENDER_YLIM,
    RENDER_ROAD_HALF, RENDER_ROAD_N,
)

# Precomputed road x-positions (relative to car)
_ROAD_X = np.linspace(-RENDER_ROAD_HALF, RENDER_ROAD_HALF, RENDER_ROAD_N)
_N_HIST = int(RENDER_HIST_SECS / DT)


# ── Environment ─────

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
    ):
        super().__init__()
        self.render_mode  = render_mode
        self.road_profile = road_profile
        self._v0          = float(vehicle_speed)
        self._y_scale     = int(render_y_scale)

        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._ode  = QuarterCarODE(physics_params)
        self._road = RoadGenerator(road_profile, vehicle_speed, road_params)
        self._rcfg = reward_config or RewardConfig()

        self._state          = self._ode.reset(self._v0)
        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._travel_sq      = 0.0
        self._last_F         = 0.0
        self._last_z_B_ddot  = 0.0
        self._episode_reward = 0.0

        self._fig            = None
        self._ren_hist       = None
        self._fd_arrow_patch = None

    # ── Gymnasium interface ─────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self._road.reset(seed=int(rng.integers(0, 2**31)))

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
        self._ren_hist       = None

        return self._obs(), self._info(0.0)

    def step(self, action):
        F_act = float(np.clip(action[0], -1.0, 1.0)) * F_MAX
        self._last_F = F_act

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

        reward = compute_reward(z_B_ddot, travel, tyre_defl, F_act, self._rcfg)
        self._episode_reward += reward

        truncated  = bool(abs(travel) > TRUNC_TRAVEL or abs(float(new_state[5])) > TRUNC_ZS)
        terminated = False

        if self._step_count >= EPISODE_STEPS and not truncated:
            terminated = True
            rms = np.sqrt(self._accel_sq / self._step_count)
            reward += compute_terminal_bonus(rms, self._rcfg)

        if self.render_mode == 'human':
            self.render()

        return self._obs(), reward, terminated, truncated, self._info(z_B_ddot)

    # ── Observation ───

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
            zeta_dot,     # 5: ζ̇
            float(x[2]),  # 6: suspension travel
            float(x[0]),  # 7: tyre deflection
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

    def get_comfort_metric(self) -> float:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        return float(max(0.0, 1.0 - rms / self._rcfg.a_limit))
    
    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    # ── Render internals ──────

    def _init_render(self):
        """Build figure and all artists once; subsequent frames only call set_data."""
        import matplotlib
        if not os.environ.get('DISPLAY'):
            matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle, Polygon

        self._ren_hist = {
            't':        collections.deque(maxlen=_N_HIST),
            'z_B':      collections.deque(maxlen=_N_HIST),
            'z_W':      collections.deque(maxlen=_N_HIST),
            'z_B_ddot': collections.deque(maxlen=_N_HIST),
            'F':        collections.deque(maxlen=_N_HIST),
            's_dot':    collections.deque(maxlen=_N_HIST),
        }

        fig = plt.figure(figsize=(14, 7))
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 2],
                       left=0.06, right=0.97, bottom=0.09, top=0.94, wspace=0.38)
        ax_s = fig.add_subplot(gs[0, 0])
        gs_r = gs[0, 1].subgridspec(4, 1, hspace=0.08)
        ax_r = [fig.add_subplot(gs_r[i]) for i in range(4)]

        # --- schematic axis ---
        ax_s.set_xlim(RENDER_XLIM)
        ax_s.set_ylim(RENDER_YLIM)
        ax_s.set_xlabel('x relative to car (m)', fontsize=9)
        ax_s.set_ylabel(f'height  (m × {self._y_scale})', fontsize=9)
        ax_s.set_title('Quarter-Car Schematic', fontsize=10)
        ax_s.tick_params(labelsize=8)

        road_poly = Polygon(np.zeros((4, 2)), closed=True,
                            fc='#c8a46e', ec='none', hatch='////', zorder=1)
        ax_s.add_patch(road_poly)

        road_line, = ax_s.plot([], [], 'k-', lw=2, zorder=2)

        tire_spring,  = ax_s.plot([], [], 'k-', lw=1.5, zorder=4)
        tire_dashpot, = ax_s.plot([], [], 'k-', lw=1.5, zorder=4)

        mw_patch = Rectangle((-RENDER_W_MW / 2, RENDER_Y_W_NOM - RENDER_H_MW / 2),
                              RENDER_W_MW, RENDER_H_MW,
                              fc='#d0d0d0', ec='black', lw=1.5, zorder=5)
        ax_s.add_patch(mw_patch)
        mw_label = ax_s.text(0, RENDER_Y_W_NOM, r'$m_W$',
                             ha='center', va='center', fontsize=10,
                             fontweight='bold', zorder=6)

        # suspension spring color changes with compression state (set in _update_artists)
        susp_spring,  = ax_s.plot([], [], '-',  lw=1.5, color='#666666', zorder=4)
        susp_dashpot, = ax_s.plot([], [], 'k-', lw=1.5, zorder=4)

        mb_patch = Rectangle((-RENDER_W_MB / 2, RENDER_Y_B_NOM - RENDER_H_MB / 2),
                              RENDER_W_MB, RENDER_H_MB,
                              fc='#aac8e8', ec='black', lw=1.5, zorder=5)
        ax_s.add_patch(mb_patch)
        mb_label = ax_s.text(0, RENDER_Y_B_NOM, r'$m_B$',
                             ha='center', va='center', fontsize=11,
                             fontweight='bold', zorder=6)

        # dashed lines at static-equilibrium heights so deflection is visible
        ax_s.axhline(RENDER_Y_W_NOM, color='#4444cc', ls='--', lw=0.8, alpha=0.45, zorder=3)
        ax_s.axhline(RENDER_Y_B_NOM, color='#cc4444', ls='--', lw=0.8, alpha=0.45, zorder=3)

        fd_text = ax_s.text(0, RENDER_Y_B_NOM, '',
                            ha='left', va='center', fontsize=9,
                            fontweight='bold', zorder=7)

        status_text = ax_s.text(
            0.02, 0.98, '', transform=ax_s.transAxes,
            va='top', ha='left', fontsize=8, family='monospace',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=3),
            zorder=8,
        )
        ax_s.text(0.98, 0.02, f'y ×{self._y_scale}',
                  transform=ax_s.transAxes,
                  va='bottom', ha='right', fontsize=7, color='#888888', zorder=8)

        # --- time-series axes ---
        ts = {}
        ts['z_B'], = ax_r[0].plot([], [], 'b-',  lw=1, label=r'$z_B$')
        ts['z_W'], = ax_r[0].plot([], [], 'r--', lw=1, label=r'$z_W$')
        ax_r[0].legend(fontsize=7, loc='upper left', framealpha=0.6)
        ax_r[0].set_ylabel('z (m)', fontsize=8)

        ts['z_B_ddot'], = ax_r[1].plot([], [], 'k-', lw=1)
        ax_r[1].set_ylabel(r'$\ddot{z}_B$ (m/s²)', fontsize=8)
        ax_r[1].axhline(0, color='gray', lw=0.5)

        ts['F'], = ax_r[2].plot([], [], color='#008800', lw=1)
        ax_r[2].set_ylabel(r'$F_D$ (N)', fontsize=8)
        ax_r[2].axhline(0, color='gray', lw=0.5)

        ts['s_dot'], = ax_r[3].plot([], [], color='#aa00aa', lw=1)
        ax_r[3].set_ylabel(r'$\dot{s}$ (m/s)', fontsize=8)
        ax_r[3].set_xlabel('t (s)', fontsize=8)

        for ax in ax_r[:-1]:
            ax.tick_params(labelbottom=False)
        for ax in ax_r:
            ax.tick_params(labelsize=7)
            ax.grid(True, lw=0.3, alpha=0.5)

        if self.render_mode == 'human':
            plt.ion()
            plt.show(block=False)

        self._fig  = fig
        self._ax_s = ax_s
        self._ax_r = ax_r
        self._artists = {
            'road_poly':    road_poly,
            'road_line':    road_line,
            'tire_spring':  tire_spring,
            'tire_dashpot': tire_dashpot,
            'mw_patch':     mw_patch,
            'mw_label':     mw_label,
            'susp_spring':  susp_spring,
            'susp_dashpot': susp_dashpot,
            'mb_patch':     mb_patch,
            'mb_label':     mb_label,
            'fd_text':      fd_text,
            'status_text':  status_text,
            'ts':           ts,
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
        h['s_dot'].append(float(x[4]))

    def _update_artists(self):
        """Reposition every artist in-place. The only allocation is the F_D arrow patch."""
        from matplotlib.patches import FancyArrow

        art = self._artists
        ys  = self._y_scale
        x   = self._state

        z_B    = float(x[5])
        z_W    = z_B + float(x[2])
        travel = float(x[2])
        F_act  = self._last_F

        y_W = RENDER_Y_W_NOM + z_W * ys
        y_B = RENDER_Y_B_NOM + z_B * ys

        # road polygon and surface contour
        t_q    = self._t + _ROAD_X / self._v0
        road_h = self._road.get_height_array(t_q) * ys
        poly_xs = np.concatenate([_ROAD_X, _ROAD_X[::-1]])
        poly_ys = np.concatenate([road_h, np.full(RENDER_ROAD_N, RENDER_YLIM[0])])
        art['road_poly'].set_xy(np.column_stack([poly_xs, poly_ys]))
        art['road_line'].set_data(_ROAD_X, road_h)

        # tire spring (left) and dashpot (right)
        y_road_0  = float(self._road.get_height(self._t)) * ys
        y_tire_top = y_W - RENDER_H_MW / 2
        art['tire_spring'].set_data(*_spring_xy(-0.15, y_tire_top, y_road_0))
        art['tire_dashpot'].set_data(*_dashpot_xy(0.15, y_tire_top, y_road_0))

        # m_W block
        art['mw_patch'].set_xy((-RENDER_W_MW / 2, y_W - RENDER_H_MW / 2))
        art['mw_label'].set_position((0, y_W))

        # suspension spring (left) and dashpot (right)
        y_susp_bot = y_W + RENDER_H_MW / 2
        y_susp_top = y_B - RENDER_H_MB / 2
        if y_susp_top > y_susp_bot + 0.05:
            # spring color: red = compressed beyond 2 cm, blue = extended, gray = normal
            spring_color = ('#cc0000' if travel >  0.02 else
                            '#0044cc' if travel < -0.02 else '#666666')
            xs, ys_ = _spring_xy(-0.22, y_susp_top, y_susp_bot)
            art['susp_spring'].set_data(xs, ys_)
            art['susp_spring'].set_color(spring_color)
            art['susp_dashpot'].set_data(*_dashpot_xy(0.22, y_susp_top, y_susp_bot))

        # m_B block
        art['mb_patch'].set_xy((-RENDER_W_MB / 2, y_B - RENDER_H_MB / 2))
        art['mb_label'].set_position((0, y_B))

        # F_D arrow - remove old patch, add new one proportional to force magnitude
        if self._fd_arrow_patch is not None:
            self._fd_arrow_patch.remove()
            self._fd_arrow_patch = None
        arrow_len = (F_act / F_MAX) * 1.2
        if abs(arrow_len) > 0.05:
            x_start = RENDER_W_MB / 2 if F_act > 0 else -RENDER_W_MB / 2
            color   = '#008800' if F_act > 0 else '#cc0000'
            al      = abs(arrow_len)
            arr = FancyArrow(x_start, y_B, arrow_len, 0.0,
                             width=0.04, length_includes_head=True,
                             head_width=0.13, head_length=min(al * 0.25, 0.25),
                             fc=color, ec=color, zorder=7)
            self._ax_s.add_patch(arr)
            self._fd_arrow_patch = arr
            txt_x = x_start + arrow_len + (0.1 if F_act > 0 else -0.1)
            art['fd_text'].set_text(f'$F_D$={F_act:+.0f} N')
            art['fd_text'].set_position((txt_x, y_B))
            art['fd_text'].set_ha('left' if F_act > 0 else 'right')
            art['fd_text'].set_color(color)
        else:
            art['fd_text'].set_text('')

        # status text block
        s_pos = self._v0 * self._t
        art['status_text'].set_text(
            f't={self._t:6.2f} s   s={s_pos:6.1f} m   ṡ={float(x[4]):.1f} m/s\n'
            f'z_B={z_B*100:+.2f} cm   z_W={z_W*100:+.2f} cm\n'
            f'ζ={self._road.get_height(self._t)*100:.3f} cm   '
            f'F_D={F_act:+.0f} N\n'
            f'ep reward={self._episode_reward:.2f}'
        )

        # time-series
        h     = self._ren_hist
        t_arr = np.array(h['t'])
        ts    = art['ts']
        ts['z_B'].set_data(t_arr,     np.array(h['z_B']))
        ts['z_W'].set_data(t_arr,     np.array(h['z_W']))
        ts['z_B_ddot'].set_data(t_arr, np.array(h['z_B_ddot']))
        ts['F'].set_data(t_arr,        np.array(h['F']))
        ts['s_dot'].set_data(t_arr,    np.array(h['s_dot']))

        if len(t_arr) > 1:
            t_hi = t_arr[-1]
            for ax in self._ax_r:
                ax.set_xlim(t_hi - RENDER_HIST_SECS, t_hi)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

    
    # ── Render helpers  ────────────────────────────────────

    def _spring_xy(x_c: float, y_top: float, y_bot: float,
                   n: int = 6, w: float = 0.08):
        """Zigzag coil spring coords for a single Line2D."""
        n_pts = 2 * n + 2
        ys = np.linspace(y_top, y_bot, n_pts)
        xs = np.full(n_pts, x_c)
        idx = np.arange(1, n_pts - 1)
        xs[1:-1] = x_c + w * np.where(idx % 2 == 1, 1.0, -1.0)
        return xs, ys


    def _dashpot_xy(x_c: float, y_top: float, y_bot: float, w: float = 0.06):
        """Piston-cylinder dashpot coords using NaN pen-lifts for a single Line2D."""
        h    = abs(y_top - y_bot)
        y_pt = y_top - h * 0.25   # piston plate
        y_cb = y_bot + h * 0.15   # cylinder bottom
        nan  = float('nan')
        xs = [x_c,      x_c,      nan,
              x_c-w/2,  x_c+w/2,  nan,
              x_c-w/2,  x_c-w/2,
              x_c+w/2,  x_c+w/2,  nan,
              x_c-w/2,  x_c+w/2,  nan,
              x_c,      x_c]
        ys = [y_top,    y_pt,     nan,
              y_pt,     y_pt,     nan,
              y_pt,     y_cb,
              y_cb,     y_pt,     nan,
              y_cb,     y_cb,     nan,
              y_cb,     y_bot]
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

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

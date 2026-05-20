import os
import collections
from typing import Tuple

import numpy as np

from QuarterCar_env.config.env_params import DT
from QuarterCar_env.config.render_params import (
    RENDER_HIST_SECS,
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
               n: int = 8, w: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    """Zigzag coil spring; returns (xs, ys) for a single Line2D."""
    n_pts = 2 * n + 2
    ys = np.linspace(y_top, y_bot, n_pts)
    xs = np.full(n_pts, x_c)
    idx = np.arange(1, n_pts - 1)
    xs[1:-1] = x_c + w * np.where(idx % 2 == 1, 1.0, -1.0)
    return xs, ys


def _damper_xy(x_c: float, y_top: float, y_bot: float, cyl_h: float):
    """
    Open-top piston-cylinder damper (\u22a4).
    y_bot: lower mass attachment; y_top: upper mass attachment.
    A short lower rod links y_bot \u2192 cylinder base (RENDER_DA_LOWER_STEM).
    The upper rod links piston top \u2192 y_top.
    Returns (upper_rod_xy, lower_rod_xy, cyl_xy, pist_rect).
    """
    hw      = RENDER_DA_W / 2
    gap     = max(y_top - y_bot, 0.05)
    cyl_bot = y_bot + RENDER_DA_LOWER_STEM
    cyl_top = cyl_bot + cyl_h
    pist_h  = RENDER_DA_PIST_H
    pist_top = float(np.clip(
        y_bot + gap * RENDER_DA_PIST_FRAC,
        cyl_bot + pist_h + 0.005,
        cyl_top - 0.005,
    ))
    pist_bot = pist_top - pist_h
    upper_rod_xy = ([x_c, x_c], [pist_top, y_top])
    lower_rod_xy = ([x_c, x_c], [y_bot,    cyl_bot])
    cyl_xy       = ([x_c - hw, x_c - hw, x_c + hw, x_c + hw],
                    [cyl_top,  cyl_bot,   cyl_bot,   cyl_top])
    m = 0.01
    pist_rect = (x_c - hw + m, pist_bot, 2 * hw - 2 * m, pist_h)
    return upper_rod_xy, lower_rod_xy, cyl_xy, pist_rect


def _ground_symbol_xy(x_c: float, y: float, half_w: float = 0.55):
    """Horizontal ground line only."""
    return np.array([x_c - half_w, x_c + half_w]), np.array([y, y])


def _build_ts_specs(env):
    specs = []
    flags = getattr(env, "_ts_flags", {})
    if flags.get("z", False):
        specs.append(('z_B', 'z_W', 'z (m)', 'b', 'r'))
    if flags.get("z_ddot", False):
        specs.append(('z_B_ddot', None, r'$\ddot{z}_B\ (m/s^2)$', 'k', None))
    if flags.get("f", False):
        specs.append(('F', None, r'$F_D\ (N)$', '#008800', None))
    if flags.get("speed", False):
        specs.append(('s_dot', None, r'$\dot{s}\ (m/s)$', '#aa00aa', None))
    return specs




#  Public render helpers

def render_env(env):
    if env.render_mode == 'none':
        return None
    if env._ren_hist is None:
        init_render(env)
    push_history(env)
    update_artists(env)
    if env.render_mode == 'human':
        import matplotlib.pyplot as plt
        env._fig.canvas.draw_idle()
        plt.pause(1e-3)
        if getattr(env, "_freeze_render", False):
            env._freeze_render = False
            plt.show(block=True)
        return None
    env._fig.canvas.draw()
    buf = env._fig.canvas.buffer_rgba()
    w, h = env._fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return img[..., :3].copy()


def close_env(env):
    if env._fig is not None:
        import matplotlib.pyplot as plt
        plt.close(env._fig)
        env._fig = None


def init_render(env):
    """Build the figure and all artists exactly once."""
    import matplotlib
    if not os.environ.get('DISPLAY'):
        matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    ts_specs = _build_ts_specs(env)
    if env._show_ts and not ts_specs:
        env._show_ts = False

    env._ren_hist = {
        't':        collections.deque(maxlen=_N_HIST),
        'z_B':      collections.deque(maxlen=_N_HIST),
        'z_W':      collections.deque(maxlen=_N_HIST),
        'z_B_ddot': collections.deque(maxlen=_N_HIST),
        'F':        collections.deque(maxlen=_N_HIST),
        's_dot':    collections.deque(maxlen=_N_HIST),
    }

    # figure layout
    win_title = f'Quarter_Car Model : ep{env._episode_count}'
    if env._show_ts:
        fig = plt.figure(figsize=(14, 7))
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 2],
                       left=0.06, right=0.97, bottom=0.09, top=0.93, wspace=0.38)
        ax_s = fig.add_subplot(gs[0, 0])
        n_ts = len(ts_specs)
        gs_r = gs[0, 1].subgridspec(n_ts, 1, hspace=0.10)
        ax_r = [fig.add_subplot(gs_r[i]) for i in range(n_ts)]
    else:
        fig  = plt.figure(figsize=(9, 7))
        ax_s = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.93)
        ax_r = []

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(win_title)

    # schematic axis
    import matplotlib.ticker as ticker
    ax_s.set_facecolor('white')
    ax_s.set_xlim(RENDER_XLIM)
    ax_s.set_ylim(RENDER_YLIM)
    ax_s.set_xlabel('position (m)', fontsize=9)
    ax_s.set_ylabel(f'height  (m \u00d7 {env._y_scale})', fontsize=9)
    ax_s.tick_params(labelsize=8)
    ax_s.spines[['top', 'right']].set_visible(False)
    ax_s.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x + env._s_pos:.0f}')
    )

    # road profile
    road_line, = ax_s.plot([], [], '-', color=RENDER_C_ROAD, lw=1.5, zorder=2,
                           label='road profile \u03b6(x)')

    # ground symbol
    ground_sym, = ax_s.plot([], [], '-', color=RENDER_C_GROUND, lw=1.5, zorder=2)

    # contact stem + dot
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

    # m_W block
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

    # m_B block
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

    # status text
    status_text = ax_s.text(
        0.02, 0.98, '', transform=ax_s.transAxes,
        va='top', ha='left', fontsize=7.5, family='monospace',
        bbox=dict(facecolor='white', alpha=0.80, edgecolor='#cccccc',
                  boxstyle='round,pad=0.3'),
        zorder=9,
    )

    # exaggeration note
    ax_s.text(0.98, 0.02, f'y \u00d7{env._y_scale}',
              transform=ax_s.transAxes,
              va='bottom', ha='right', fontsize=7, color='#aaaaaa', zorder=9)

    ax_s.legend(fontsize=7, loc='upper right', framealpha=0.7,
                handlelength=1.5, borderpad=0.4)

    # time-series axes
    _ts_specs = ts_specs
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

    if env.render_mode == 'human':
        plt.ion()
        plt.show(block=False)

    env._fig  = fig
    env._ax_s = ax_s
    env._ax_r = ax_r
    env._artists = {
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
        'status_text':    status_text,
        'ts':             ts,
    }


def push_history(env):
    x   = env._state
    z_B = float(x[5])
    z_W = z_B + float(x[2])
    h   = env._ren_hist
    h['t'].append(env._t)
    h['z_B'].append(z_B)
    h['z_W'].append(z_W)
    h['z_B_ddot'].append(env._last_z_B_ddot)
    h['F'].append(0.0)
    h['s_dot'].append(float(env._v))


def update_artists(env):
    """Update all artists in-place."""

    art = env._artists
    ys  = env._y_scale
    x   = env._state

    z_B    = float(x[5])
    z_W    = z_B + float(x[2])
    zeta_0 = float(env._road.get_height(env._t))

    # draw-space heights for the two masses (RENDER_GROUND_Y shifts entire system)
    y_W      = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_W * ys
    y_B      = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_B * ys
    y_road_0 = RENDER_GROUND_Y + zeta_0 * ys

    # road profile (gray line, car at x=0, road scrolls left)
    v_disp  = max(env._v, 0.1)
    t_q     = env._t + _ROAD_X / v_disp
    road_h  = env._road.get_height_array(t_q) * ys + RENDER_GROUND_Y
    art['road_line'].set_data(_ROAD_X, road_h)

    h = env._ren_hist

    y_road_0 += Y_LINE_OFFSET
    # ground symbol + contact stem + dot
    gx, gy = _ground_symbol_xy(0.0, y_road_0, half_w=RENDER_W_MB / 2 + 0.15)
    art['ground_sym'].set_data(gx, gy)
    art['contact_stem'].set_data([0.0, 0.0], [y_road_0 - Y_LINE_OFFSET + RENDER_CONTACT_STEM, y_road_0])
    art['contact_dot'].set_data([0.0], [y_road_0 - Y_LINE_OFFSET])

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

    # status text
    art['status_text'].set_text(
        f't={env._t:6.2f} s    s={env._s_pos:6.1f} m\n'
        f'z_B={z_B*100:+.2f} cm  z_W={z_W*100:+.2f} cm\n'
        f'\u03b6={zeta_0*100:.3f} cm\n'
        f'v={env._v:.1f} m/s    '
        f'ep reward={env._episode_reward:.2f}'
    )

    # time-series
    if not env._show_ts:
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
        for ax in env._ax_r:
            ax.set_xlim(t_hi - RENDER_HIST_SECS, t_hi)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

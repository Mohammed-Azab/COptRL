# evaluation runner for the human driver baseline
from __future__ import annotations

import os

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src' / 'gym_env'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'src' / 'baseline'))
sys.path.insert(0, str(_ROOT / 'src' / 'eval'))

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.config.reward_params import load_reward_config
from QuarterCar_env.config.env_params import DT
from driver import HumanDriverController
from scenario_loader import load_scenario, list_scenarios, make_road_generator
from data_logger import RunLogger, EpisodeData


def _load_driver_cfg() -> dict:
    p = _ROOT / 'config' / 'baseline' / 'human_driver_params.yaml'
    with open(p) as fh:
        return yaml.safe_load(fh)


def _steps_for_road(road, dt: float = 0.02, margin: float = 1.5) -> int:
    # enough steps to cross the full road at the design speed
    if road is None or not hasattr(road, '_bumps') or not road._bumps:
        return 1200
    x0, _, L = road._bumps[-1]
    road_length = x0 + L + 20.0          # last bump end + generous buffer
    v = max(float(road.speed), 1.0)
    return max(int(road_length / v / dt * margin), 600)


def run_episode(env, ctrl: HumanDriverController, seed: int,
                scenario_road=None, collect_frames: bool = False,
                render_live: bool = False) -> dict:
    # deep-copy so the env's set_speed() calls don't corrupt the shared template
    road_for_ep = copy.deepcopy(scenario_road) if scenario_road is not None else None

    # ensure enough steps to traverse all bumps (default budget assumes v_max)
    raw = env.unwrapped
    raw._max_episode_steps = _steps_for_road(road_for_ep)

    opts   = {"road": road_for_ep} if road_for_ep is not None else {}
    obs, _ = env.reset(seed=seed, options=opts or None)

    # respect the road's design speed (scenarios run at their specified km/h)
    saved_v_max = ctrl.v_max
    if road_for_ep is not None and hasattr(road_for_ep, 'speed'):
        ctrl.v_max = min(ctrl.v_max, float(road_for_ep.speed))

    ep_return  = 0.0
    accel_sq   = 0.0
    n_steps    = 0
    act_times: list[float] = []
    speeds:    list[float] = []
    v_refs:    list[float] = []
    accels:    list[float] = []
    actions:   list[float] = []
    frames:    list        = []
    done       = False

    while not done:
        if render_live:
            env.render()
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        t0 = time.perf_counter()
        u  = ctrl.act(raw._state.copy(), raw._s_pos, raw._road)
        act_times.append(time.perf_counter() - t0)

        _, reward, terminated, truncated, info = env.step(
            np.array([u], dtype=np.float32)
        )
        ep_return += reward
        accel_sq  += info.get('z_B_ddot', 0.0) ** 2
        n_steps   += 1
        speeds.append(info.get('speed',    0.0))
        v_refs.append(info.get('v_ref',    0.0))
        accels.append(info.get('z_B_ddot', 0.0))
        actions.append(float(u))
        done = terminated or truncated

    ctrl.v_max = saved_v_max  # restore after episode

    rms_accel = float(np.sqrt(accel_sq / max(n_steps, 1)))
    cfg       = load_reward_config()
    return {
        'episode_return': round(float(ep_return),   2),
        'rms_accel':      round(rms_accel,           4),
        'comfort_score':  round(max(0.0, 1.0 - rms_accel / cfg.a_limit), 4),
        'n_steps':        n_steps,
        'bumps_passed':   int(raw._bumps_passed),
        'bumps_total':    len(raw._bump_ends),
        'act_us_mean':    round(float(np.mean(act_times)) * 1e6, 1),
        '_speeds':        speeds,
        '_v_refs':        v_refs,
        '_accels':        accels,
        '_actions':       actions,
        '_frames':        frames,
    }


def _save_gif(frames: list, path: Path) -> None:
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     optimize=False, duration=20, loop=0)
        print(f'  GIF saved → {path}')
    except ImportError:
        print('  [PIL not installed — cannot save GIF; run: pip install Pillow]')


def _save_plot(result: dict, ep_i: int, save_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        from QuarterCar_env.config.env_params import DT
    except ImportError:
        DT = 0.01

    t = np.arange(result['n_steps']) * DT

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, [v * 3.6 for v in result['_speeds']], label='speed km/h')
    axes[0].plot(t, [v * 3.6 for v in result['_v_refs']], '--', label='v_ref km/h', alpha=0.6)
    axes[0].set_ylabel('speed [km/h]')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, result['_accels'], color='tab:orange', label='body accel')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_ylabel('z̈_B [m/s²]')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, result['_actions'], color='tab:green', label='action u')
    axes[2].axhline(0, color='gray', lw=0.7)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_ylabel('action [-1,1]')
    axes[2].set_xlabel('time [s]')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f'Human Driver — ep {ep_i+1}  return={result["episode_return"]:.1f}',
        fontsize=11,
    )
    fig.tight_layout()
    out = save_dir / f'human_driver_ep{ep_i+1}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  plot saved → {out}')


def main() -> None:
    dcfg = _load_driver_cfg()

    ap = argparse.ArgumentParser('Rule-based human driver baseline')
    ap.add_argument('--n-episodes',      type=int,   default=20)
    ap.add_argument('--seed',            type=int,   default=42)
    ap.add_argument('--road',            default='speed_bump')
    ap.add_argument('--scenario',        default=None,
                    help=f'fixed eval scenario ({list_scenarios()})')
    ap.add_argument('--preview',         type=float,
                    default=dcfg.get('preview_m', 40.0))
    ap.add_argument('--zeta-dot-limit',  type=float,
                    default=dcfg.get('zeta_dot_limit', 1.5))
    ap.add_argument('--render',           action='store_true',
                    help='show live window (render_mode=human, like eval.py)')
    ap.add_argument('--save-gif',        action='store_true',
                    help='save per-episode GIF (headless-safe offscreen rendering)')
    ap.add_argument('--save-plots',      action='store_true',
                    help='save per-episode time-series PNG')
    ap.add_argument('--results-dir',     default=None)
    ap.add_argument('--out',             default=None)
    ap.add_argument('--log-data',        metavar='DIR', default=None,
                    help='Save run data (.mat/.npz) to DIR/HumanDriver/<road>/<ts>/. '
                         'Omit DIR to use <repo>/data/.',
                    nargs='?', const='__default__')
    args = ap.parse_args()

    if args.save_gif and not args.render:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    cfg  = load_reward_config()
    ctrl = HumanDriverController(
        v_max          = float(cfg.v_max),
        v_min          = float(cfg.v_min),
        a_max          = float(cfg.a_max),
        preview_m      = args.preview,
        zeta_dot_limit = args.zeta_dot_limit,
        a_brake        = float(dcfg.get('a_brake',      2.5)),
        a_accel        = float(dcfg.get('a_accel',      2.0)),
        ctrl_horizon   = float(dcfg.get('ctrl_horizon', 1.2)),
    )

    if args.render:
        render_mode = 'human'
    elif args.save_gif:
        render_mode = 'rgb_array'
    else:
        render_mode = 'none'
    env = gym.make('QuarterCar_env/QuarterCar', road_profile=args.road,
                   render_mode=render_mode)

    scenario_road  = None
    scenario_label = args.road
    if args.scenario:
        _, _, sc_name, sc_desc = load_scenario(args.scenario)
        scenario_road  = make_road_generator(args.scenario)
        scenario_label = args.scenario
        print(f'\n  Scenario : {sc_name}')
        print(f'           : {sc_desc}')

    save_dir = (
        Path(args.results_dir) if args.results_dir
        else _ROOT / 'eval' / 'results' / 'human_driver' / f'{scenario_label}'
    )

    print(f'\n  Human Driver Baseline  (rule-based)')
    print(f'  preview   : {args.preview} m        (config/baseline/human_driver_params.yaml)')
    print(f'  ζ̇ limit  : {args.zeta_dot_limit} m/s')
    print(f'  a_brake   : {ctrl.a_brake} m/s²')
    print(f'  episodes  : {args.n_episodes}')
    print(f'  road      : {scenario_label}\n')
    print(f'  {"Ep":>3}  {"Return":>9}  {"RMS-a m/s²":>10}  {"Comfort":>8}  {"Bumps":>7}  {"Act µs":>8}')
    print(f'  {"-"*3}  {"-"*9}  {"-"*10}  {"-"*8}  {"-"*7}  {"-"*8}')

    log_root = (
        _ROOT / 'data'
        if args.log_data == '__default__'
        else Path(args.log_data) if args.log_data
        else None
    )
    run_logger: RunLogger | None = (
        RunLogger(
            method='HumanDriver',
            road=scenario_label,
            out_root=log_root,
            dt=DT,
            v_max_kmh=cfg.v_max * 3.6,
            a_comfort=cfg.a_comfort,
            a_limit=cfg.a_limit,
        )
        if log_root is not None else None
    )

    results: list[dict] = []
    for ep in range(args.n_episodes):
        r = run_episode(env, ctrl, seed=args.seed + ep,
                        scenario_road=scenario_road,
                        collect_frames=args.save_gif,
                        render_live=args.render)
        results.append(r)
        bumps_str = f'{r["bumps_passed"]}/{r["bumps_total"]}'
        print(
            f'  {ep+1:>3}  {r["episode_return"]:>+9.1f}  '
            f'{r["rms_accel"]:>10.3f}  '
            f'{r["comfort_score"]:>8.3f}  '
            f'{bumps_str:>7}  '
            f'{r["act_us_mean"]:>8.1f}'
        )
        if run_logger is not None:
            run_logger.add(EpisodeData(
                v=r['_speeds'],
                v_ref=r['_v_refs'],
                z_B_ddot=r['_accels'],
                action=r['_actions'],
                episode_return=r['episode_return'],
                rms_accel=r['rms_accel'],
                comfort_score=r['comfort_score'],
            ))
        if args.save_plots or args.save_gif:
            save_dir.mkdir(parents=True, exist_ok=True)
        if args.save_plots:
            _save_plot(r, ep, save_dir)
        if args.save_gif and r['_frames']:
            _save_gif(r['_frames'], save_dir / f'human_driver_ep{ep+1}.gif')

    env.close()

    rets     = [r['episode_return']  for r in results]
    rms_vals = [r['rms_accel']       for r in results]
    comforts = [r['comfort_score']   for r in results]

    print(f'\n  {"avg":>3}  {np.mean(rets):>+9.1f}  {np.mean(rms_vals):>10.3f}  {np.mean(comforts):>8.3f}')
    print(f'  {"std":>3}  {np.std(rets):>9.1f}  {np.std(rms_vals):>10.3f}  {np.std(comforts):>8.3f}')
    print(f'  {"max":>3}  {np.max(rets):>+9.1f}')
    print(f'  {"min":>3}  {np.min(rets):>+9.1f}')

    # strip internal time-series before saving JSON
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    summary = {
        'method':           'HumanDriver-rule-based',
        'preview_m':        args.preview,
        'zeta_dot_limit':   args.zeta_dot_limit,
        'a_brake':          ctrl.a_brake,
        'n_episodes':       args.n_episodes,
        'mean_return':      round(float(np.mean(rets)),     2),
        'std_return':       round(float(np.std(rets)),      2),
        'max_return':       round(float(np.max(rets)),      2),
        'min_return':       round(float(np.min(rets)),      2),
        'mean_rms_accel':   round(float(np.mean(rms_vals)), 4),
        'mean_comfort':     round(float(np.mean(comforts)), 4),
        'episodes':         clean,
    }

    out = Path(args.out) if args.out else save_dir / 'summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\n  Results → {out}')

    if run_logger is not None:
        saved = run_logger.save()
        print(f'  data logged → {run_logger.save_dir}')
        if 'mat' not in saved:
            print('  (install scipy for .mat output: pip install scipy)')


if __name__ == '__main__':
    main()

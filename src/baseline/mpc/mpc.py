from __future__ import annotations

import os

from pathlib import Path as _Path

# ACADOS_SOURCE_DIR must be the cmake install prefix (~/acados) which has
# include/acados_c/ and lib/libacados.so — source tree lacks both
_acados = _Path.home() / 'acados'
os.environ['ACADOS_SOURCE_DIR'] = str(_acados)
_lib = str(_acados / 'lib')
if _lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # baseline/mpc on path for controller/ocp

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.config.env_params import PHYSICS
from QuarterCar_env.config.reward_params import load_reward_config
from controller import MPCController
from scenario_loader import load_scenario, list_scenarios, make_road_generator


def _load_mpc_cfg() -> dict:
    p = _ROOT / 'config' / 'baseline' / 'mpc_params.yaml'
    with open(p) as fh:
        return yaml.safe_load(fh)


def run_episode(env, ctrl: MPCController, seed: int,
                scenario_road=None, collect_frames: bool = False,
                render_live: bool = False) -> dict:
    # deep-copy so the env's set_speed() calls don't corrupt the shared template
    road_for_ep = copy.deepcopy(scenario_road) if scenario_road is not None else None
    opts     = {"road": road_for_ep} if road_for_ep is not None else {}
    obs, _   = env.reset(seed=seed, options=opts or None)
    raw      = env.unwrapped

    # respect road's design speed (scenario speed cap)
    saved_v_max = ctrl._cfg.v_max
    if road_for_ep is not None and hasattr(road_for_ep, 'speed'):
        ctrl._cfg = ctrl._cfg._replace(v_max=min(ctrl._cfg.v_max, float(road_for_ep.speed))) \
            if hasattr(ctrl._cfg, '_replace') else ctrl._cfg
    ctrl.reset(raw._road)

    ep_return  = 0.0
    accel_sq   = 0.0
    n_steps    = 0
    solve_times: list[float] = []
    speeds:     list[float] = []
    v_refs:     list[float] = []
    accels:     list[float] = []
    actions:    list[float] = []
    frames:     list        = []
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
        solve_times.append(time.perf_counter() - t0)

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
        done       = terminated or truncated

    rms_accel = float(np.sqrt(accel_sq / max(n_steps, 1)))
    cfg       = load_reward_config()
    return {
        'episode_return': round(float(ep_return),   2),
        'rms_accel':      round(rms_accel,           4),
        'comfort_score':  round(max(0.0, 1.0 - rms_accel / cfg.a_limit), 4),
        'n_steps':        n_steps,
        'bumps_passed':   int(raw._bumps_passed),
        'solve_ms_mean':  round(float(np.mean(solve_times)) * 1e3, 2),
        'solve_ms_max':   round(float(np.max(solve_times))  * 1e3, 2),
        '_speeds':        speeds,
        '_v_refs':        v_refs,
        '_accels':        accels,
        '_actions':       actions,
        '_frames':        frames,
    }


def _best_rl(models_dir: Path) -> dict | None:
    best = None
    for p in models_dir.rglob('summary.json'):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        ec = s.get('eval_curve') or {}
        ret = ec.get('best_mean_return')
        if ret is not None:
            if best is None or ret > best.get('best_mean_return', -1e9):
                best = {
                    'exp':              p.parent.name,
                    'algo':             s.get('config', {}).get('algo', '?'),
                    'best_mean_return': ret,
                    'best_step':        ec.get('best_step'),
                    'timesteps':        s.get('config', {}).get('timesteps'),
                }
    return best


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
        DT = 0.02

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
        f'MPC — ep {ep_i+1}  return={result["episode_return"]:.1f}  '
        f'solve={result["solve_ms_mean"]:.1f}ms',
        fontsize=11,
    )
    fig.tight_layout()
    out = save_dir / f'mpc_ep{ep_i+1}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  plot saved → {out}')


def main() -> None:
    mcfg = _load_mpc_cfg()

    ap = argparse.ArgumentParser('MPC baseline for QuarterCar Model')
    ap.add_argument('--n-episodes', type=int,   default=mcfg.get('n_episodes', 20))
    ap.add_argument('--horizon',    type=int,   default=mcfg.get('N', 50),
                    help='prediction horizon (steps)')
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--road',       default='speed_bump')
    ap.add_argument('--scenario',   default=None,
                    help=f'fixed eval scenario name (available: {list_scenarios()})')
    ap.add_argument('--render',      action='store_true',
                    help='show live window (render_mode=human, like eval.py)')
    ap.add_argument('--save-gif',   action='store_true',
                    help='save per-episode GIF (headless-safe offscreen rendering)')
    ap.add_argument('--save-plots', action='store_true',
                    help='save per-episode time-series PNG')
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--out',        default=None)
    args = ap.parse_args()

    if args.save_gif and not args.render:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    cfg  = load_reward_config()
    ctrl = MPCController(
        cfg=cfg, physics=dict(PHYSICS), N=args.horizon,
        nlp_solver_max_iter=mcfg.get('nlp_solver_max_iter', 10),
    )

    if args.render:
        render_mode = 'human'
    elif args.save_gif:
        render_mode = 'rgb_array'
    else:
        render_mode = 'none'
    env  = gym.make('QuarterCar_env/QuarterCar', road_profile=args.road,
                    render_mode=render_mode)

    scenario_road = None
    scenario_label = args.road
    if args.scenario:
        _, speed, sc_name, sc_desc = load_scenario(args.scenario)
        scenario_road  = make_road_generator(args.scenario)
        scenario_label = args.scenario
        print(f'\n  Scenario : {sc_name}')
        print(f'           : {sc_desc}')
        print(f'           : v={speed*3.6:.1f} km/h, {len(scenario_road._bumps)} bump(s)')

    save_dir = (
        Path(args.results_dir) if args.results_dir
        else _ROOT / 'eval' / 'results' / f'mpc_{scenario_label}'
    )

    print(f'\n  MPC Baseline')
    print(f'  horizon  : {args.horizon} steps  ({args.horizon * 0.02:.2f}s)  '
          f'(config/baseline/mpc_params.yaml)')
    print(f'  episodes : {args.n_episodes}')
    print(f'  road     : {scenario_label}')
    print(f'  solver   : acados SQP-RTI + HPIPM\n')
    print(f'  {"Ep":>3}  {"Return":>9}  {"RMS-a m/s²":>10}  {"Comfort":>8}  '
          f'{"Bumps":>5}  {"Solve ms":>9}')
    print(f'  {"-"*3}  {"-"*9}  {"-"*10}  {"-"*8}  {"-"*5}  {"-"*9}')

    results: list[dict] = []
    for ep in range(args.n_episodes):
        r = run_episode(env, ctrl, seed=args.seed + ep,
                        scenario_road=scenario_road,
                        collect_frames=args.save_gif,
                        render_live=args.render)
        results.append(r)
        print(
            f'  {ep+1:>3}  {r["episode_return"]:>+9.1f}  '
            f'{r["rms_accel"]:>10.3f}  '
            f'{r["comfort_score"]:>8.3f}  '
            f'{r["bumps_passed"]:>5}  '
            f'{r["solve_ms_mean"]:>9.2f}'
        )
        if args.save_plots or args.save_gif:
            save_dir.mkdir(parents=True, exist_ok=True)
        if args.save_plots:
            _save_plot(r, ep, save_dir)
        if args.save_gif and r['_frames']:
            _save_gif(r['_frames'], save_dir / f'mpc_ep{ep+1}.gif')

    env.close()

    rets     = [r['episode_return']  for r in results]
    rms_vals = [r['rms_accel']       for r in results]
    comforts = [r['comfort_score']   for r in results]
    solves   = [r['solve_ms_mean']   for r in results]

    print(f'\n  {"avg":>3}  {np.mean(rets):>+9.1f}  {np.mean(rms_vals):>10.3f}  '
          f'{np.mean(comforts):>8.3f}  {"":>5}  {np.mean(solves):>9.2f}')
    print(f'  {"std":>3}  {np.std(rets):>9.1f}  {np.std(rms_vals):>10.3f}  '
          f'{np.std(comforts):>8.3f}')
    print(f'  {"max":>3}  {np.max(rets):>+9.1f}')
    print(f'  {"min":>3}  {np.min(rets):>+9.1f}')

    best_rl = _best_rl(_ROOT / 'models')
    if best_rl:
        gap = np.mean(rets) - best_rl['best_mean_return']
        print(f'\n  RL best  : {best_rl["best_mean_return"]:+.1f}  '
              f'({best_rl["exp"]} @ step {best_rl["best_step"]:,})')
        print(f'  MPC mean : {np.mean(rets):+.1f}')
        print(f'  gap      : {gap:+.1f}  (positive = MPC better)')

    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    summary = {
        'method':          'MPC-acados',
        'horizon_steps':   args.horizon,
        'horizon_secs':    round(args.horizon * 0.02, 3),
        'n_episodes':      args.n_episodes,
        'mean_return':     round(float(np.mean(rets)),     2),
        'std_return':      round(float(np.std(rets)),      2),
        'max_return':      round(float(np.max(rets)),      2),
        'min_return':      round(float(np.min(rets)),      2),
        'mean_rms_accel':  round(float(np.mean(rms_vals)), 4),
        'mean_comfort':    round(float(np.mean(comforts)), 4),
        'mean_solve_ms':   round(float(np.mean(solves)),   3),
        'rl_comparison':   best_rl,
        'episodes':        clean,
    }

    out = Path(args.out) if args.out else save_dir / 'summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\n  Results → {out}')


if __name__ == '__main__':
    main()

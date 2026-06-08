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
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src' / 'gym_env'))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # baseline/mpc on path for controller/ocp

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.config.env_params import PHYSICS
from QuarterCar_env.config.reward_params import load_reward_config
from controller import MPCController
from scenario_loader import load_scenario, list_scenarios, make_road_generator


def run_episode(env, ctrl: MPCController, seed: int,
                scenario_road=None) -> dict:
    opts     = {"road": scenario_road} if scenario_road is not None else {}
    obs, _   = env.reset(seed=seed, options=opts or None)
    raw      = env.unwrapped
    if scenario_road is not None:
        raw._road = scenario_road   # ensure MPC solver sees the same road
    ctrl.reset(raw._road)

    ep_return  = 0.0
    accel_sq   = 0.0
    n_steps    = 0
    solve_times: list[float] = []
    done       = False

    while not done:
        t0 = time.perf_counter()
        u  = ctrl.act(raw._state.copy(), raw._s_pos, raw._road)
        solve_times.append(time.perf_counter() - t0)

        _, reward, terminated, truncated, info = env.step(
            np.array([u], dtype=np.float32)
        )
        ep_return += reward
        accel_sq  += info.get('z_B_ddot', 0.0) ** 2
        n_steps   += 1
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
    }


def _best_rl(models_dir: Path) -> dict | None:
    # scan all summary.json files, return best eval_curve.best_mean_return
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


def main() -> None:
    ap = argparse.ArgumentParser('MPC baseline for QuarterCar Model')
    ap.add_argument('--n-episodes', type=int,   default=20)
    ap.add_argument('--horizon',    type=int,   default=50,   help='prediction horizon (steps)')
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--road',       default='speed_bump')
    ap.add_argument('--scenario',   default=None,
                    help=f'fixed eval scenario name (available: {list_scenarios()})')
    ap.add_argument('--out',        default=None)
    args = ap.parse_args()

    cfg  = load_reward_config()
    ctrl = MPCController(cfg=cfg, physics=dict(PHYSICS), N=args.horizon)
    env  = gym.make('QuarterCar_env/QuarterCar', road_profile=args.road)

    scenario_road = None
    scenario_label = args.road
    if args.scenario:
        _, speed, sc_name, sc_desc = load_scenario(args.scenario)
        scenario_road  = make_road_generator(args.scenario)
        scenario_label = args.scenario
        print(f'\n  Scenario : {sc_name}')
        print(f'           : {sc_desc}')
        print(f'           : v={speed*3.6:.1f} km/h, {len(scenario_road._bumps)} bump(s)')

    print(f'\n  MPC Baseline')
    print(f'  horizon  : {args.horizon} steps  ({args.horizon * 0.02:.2f}s lookahead)')
    print(f'  episodes : {args.n_episodes}')
    print(f'  road     : {scenario_label}')
    print(f'  solver   : acados SQP-RTI + HPIPM\n')
    print(f'  {"Ep":>3}  {"Return":>9}  {"RMS-a m/s²":>10}  {"Comfort":>8}  {"Bumps":>5}  {"Solve ms":>9}')
    print(f'  {"-"*3}  {"-"*9}  {"-"*10}  {"-"*8}  {"-"*5}  {"-"*9}')

    results: list[dict] = []
    for ep in range(args.n_episodes):
        r = run_episode(env, ctrl, seed=args.seed + ep, scenario_road=scenario_road)
        results.append(r)
        print(
            f'  {ep+1:>3}  {r["episode_return"]:>+9.1f}  '
            f'{r["rms_accel"]:>10.3f}  '
            f'{r["comfort_score"]:>8.3f}  '
            f'{r["bumps_passed"]:>5}  '
            f'{r["solve_ms_mean"]:>9.2f}'
        )

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
        'episodes':        results,
    }

    out = Path(args.out) if args.out else _ROOT / 'eval' / 'results' / 'mpc_baseline.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\n  Results → {out}')


if __name__ == '__main__':
    main()

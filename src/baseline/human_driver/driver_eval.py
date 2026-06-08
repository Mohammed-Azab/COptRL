"""Evaluation harness for the rule-based human driver baseline."""
from __future__ import annotations

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
sys.path.insert(0, str(_ROOT / 'src' / 'baseline'))

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.config.reward_params import load_reward_config
from driver import HumanDriverController
from scenario_loader import load_scenario, list_scenarios, make_road_generator


def run_episode(env, ctrl: HumanDriverController, seed: int,
                scenario_road=None) -> dict:
    opts   = {"road": scenario_road} if scenario_road is not None else {}
    obs, _ = env.reset(seed=seed, options=opts or None)
    raw    = env.unwrapped

    ep_return  = 0.0
    accel_sq   = 0.0
    n_steps    = 0
    act_times: list[float] = []
    done       = False

    while not done:
        t0 = time.perf_counter()
        u  = ctrl.act(raw._state.copy(), raw._s_pos, raw._road)
        act_times.append(time.perf_counter() - t0)

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
        'act_us_mean':    round(float(np.mean(act_times)) * 1e6, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser('Rule-based human driver baseline')
    ap.add_argument('--n-episodes',      type=int,   default=20)
    ap.add_argument('--seed',            type=int,   default=42)
    ap.add_argument('--road',            default='speed_bump')
    ap.add_argument('--scenario',        default=None,
                    help=f'fixed eval scenario ({list_scenarios()})')
    ap.add_argument('--preview',         type=float, default=40.0,
                    help='driver look-ahead distance [m]')
    ap.add_argument('--zeta-dot-limit',  type=float, default=1.5,
                    help='peak road velocity the driver tolerates [m/s]')
    ap.add_argument('--out',             default=None)
    args = ap.parse_args()

    cfg  = load_reward_config()
    ctrl = HumanDriverController(
        v_max          = float(cfg.v_max),
        v_min          = float(cfg.v_min),
        a_max          = float(cfg.a_max),
        preview_m      = args.preview,
        zeta_dot_limit = args.zeta_dot_limit,
    )
    env = gym.make('QuarterCar_env/QuarterCar', road_profile=args.road)

    scenario_road  = None
    scenario_label = args.road
    if args.scenario:
        _, speed, sc_name, sc_desc = load_scenario(args.scenario)
        scenario_road  = make_road_generator(args.scenario)
        scenario_label = args.scenario
        print(f'\n  Scenario : {sc_name}')
        print(f'           : {sc_desc}')

    print(f'\n  Human Driver Baseline  (rule-based)')
    print(f'  preview   : {args.preview} m')
    print(f'  ζ̇ limit  : {args.zeta_dot_limit} m/s  (crossing comfort)')
    print(f'  episodes  : {args.n_episodes}')
    print(f'  road      : {scenario_label}\n')
    print(f'  {"Ep":>3}  {"Return":>9}  {"RMS-a m/s²":>10}  {"Comfort":>8}  {"Bumps":>5}  {"Act µs":>8}')
    print(f'  {"-"*3}  {"-"*9}  {"-"*10}  {"-"*8}  {"-"*5}  {"-"*8}')

    results: list[dict] = []
    for ep in range(args.n_episodes):
        r = run_episode(env, ctrl, seed=args.seed + ep,
                        scenario_road=scenario_road)
        results.append(r)
        print(
            f'  {ep+1:>3}  {r["episode_return"]:>+9.1f}  '
            f'{r["rms_accel"]:>10.3f}  '
            f'{r["comfort_score"]:>8.3f}  '
            f'{r["bumps_passed"]:>5}  '
            f'{r["act_us_mean"]:>8.1f}'
        )

    env.close()

    rets     = [r['episode_return']  for r in results]
    rms_vals = [r['rms_accel']       for r in results]
    comforts = [r['comfort_score']   for r in results]

    print(f'\n  {"avg":>3}  {np.mean(rets):>+9.1f}  {np.mean(rms_vals):>10.3f}  '
          f'{np.mean(comforts):>8.3f}')
    print(f'  {"std":>3}  {np.std(rets):>9.1f}  {np.std(rms_vals):>10.3f}  '
          f'{np.std(comforts):>8.3f}')
    print(f'  {"max":>3}  {np.max(rets):>+9.1f}')
    print(f'  {"min":>3}  {np.min(rets):>+9.1f}')

    summary = {
        'method':           'HumanDriver-rule-based',
        'preview_m':        args.preview,
        'zeta_dot_limit':   args.zeta_dot_limit,
        'n_episodes':       args.n_episodes,
        'mean_return':      round(float(np.mean(rets)),     2),
        'std_return':       round(float(np.std(rets)),      2),
        'max_return':       round(float(np.max(rets)),      2),
        'min_return':       round(float(np.min(rets)),      2),
        'mean_rms_accel':   round(float(np.mean(rms_vals)), 4),
        'mean_comfort':     round(float(np.mean(comforts)), 4),
        'episodes':         results,
    }

    out = Path(args.out) if args.out else _ROOT / 'eval' / 'results' / 'human_driver.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f'\n  Results → {out}')


if __name__ == '__main__':
    main()

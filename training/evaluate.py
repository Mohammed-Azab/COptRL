import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from quarter_car_core.quarter_car_env import QuarterCarEnv

_SB3_ALGOS = {'sac', 'td3', 'ppo'}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_config(config_path=None):
    path = (Path(config_path) if config_path
            else Path(__file__).parent / 'configs' / 'env_config.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def _default_model_path(algo: str) -> Path | None:
    candidates = []
    model_root = _project_root() / 'models'
    for model_dir in model_root.glob(f'{algo}_*'):
        candidates.extend([
            model_dir / 'best' / 'best_model.zip',
            model_dir / f'{algo}_final.zip',
            model_dir / f'{algo}_final',
        ])

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _resolve_model_path(algo: str, model_path: str | None) -> Path:
    if model_path:
        return Path(model_path)

    default_path = _default_model_path(algo)
    if default_path is None:
        raise FileNotFoundError(
            f'No trained model found for {algo}. Pass --model_path explicitly or train first.'
        )
    return default_path


def _load_model(algo: str, model_path: str | None, device: str = 'auto'):
    if algo not in _SB3_ALGOS:
        return None
    from stable_baselines3 import SAC, TD3, PPO
    resolved_path = _resolve_model_path(algo, model_path)
    return {'sac': SAC, 'td3': TD3, 'ppo': PPO}[algo].load(resolved_path, device=device)


def _run_episode(env: QuarterCarEnv, model, algo: str) -> tuple:
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        # passive = zero actuator force (spring/damper only) — the no-control baseline
        if algo == 'passive':
            action = np.array([0.0], dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        z_s, z_s_dot, z_u, _, z_r, _, travel, _ = obs
        records.append({
            't':      info['episode_time'],
            'z_s':    float(z_s),
            'z_u':    float(z_u),
            'z_r':    float(z_r),
            'travel': float(travel),
            'F_act':  float(action[0]) * 10_000.0,
            'reward': float(reward),
        })
        done = terminated or truncated
    return pd.DataFrame(records), info


def _plot(df_rl: pd.DataFrame, df_passive: pd.DataFrame | None, algo: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df_rl['t'], df_rl['z_s'], label=f'{algo} z_s')
    axes[0].plot(df_rl['t'], df_rl['z_r'], '--', label='road z_r')
    if df_passive is not None:
        axes[0].plot(df_passive['t'], df_passive['z_s'], ':',
                     label='passive z_s', alpha=0.7)
    axes[0].set_ylabel('Height [m]')
    axes[0].legend(fontsize=8)

    dt_med = df_rl['t'].diff().median()
    accel  = np.gradient(np.gradient(df_rl['z_s'].values, dt_med), dt_med)
    axes[1].plot(df_rl['t'], accel)
    axes[1].set_ylabel('z_s_ddot [m/s²]')

    axes[2].plot(df_rl['t'], df_rl['travel'])
    axes[2].axhline( 0.08, color='r', linestyle='--', linewidth=0.8, label='±0.08 m')
    axes[2].axhline(-0.08, color='r', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel('Susp. travel [m]')
    axes[2].legend(fontsize=8)

    axes[3].plot(df_rl['t'], df_rl['F_act'])
    axes[3].set_ylabel('F_act [N]')
    axes[3].set_xlabel('Time [s]')

    plt.suptitle(f'Evaluation — {algo}')
    plt.tight_layout()
    plt.show()


def define_args():
    env_cfg      = load_env_config()
    road_options = env_cfg['road_options']
    algorithms   = env_cfg['algorithms']

    parser = argparse.ArgumentParser(
        description='Evaluate a trained suspension controller on the Quarter Car Model'
    )
    parser.add_argument(
        '--algo',
        type=str,
        required=True,
        choices=algorithms,
        help='Algorithm used to train the model (or "passive" for no-control baseline).',
    )
    parser.add_argument(
        '--road',
        type=str,
        default='flat',
        choices=road_options,
        help='Road profile to evaluate on (default: flat).',
    )
    parser.add_argument(
        '--model_path',
        '--model',
        dest='model_path',
        type=str,
        default=None,
        help='Path to a model .zip file. Defaults to the newest trained model for the selected algo.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Number of evaluation episodes (default: 3).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42).',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show trajectory plots after evaluation.',
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save per-episode CSVs to results/.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Inference device (default: auto).',
    )
    return parser.parse_args()


def main():
    args = define_args()

    model = None if args.algo == 'passive' else _load_model(args.algo, args.model_path, args.device)
    env   = QuarterCarEnv(road_profile=args.road)

    all_dfs, all_info = [], []
    for _ in range(args.episodes):
        df, info = _run_episode(env, model, args.algo)
        all_dfs.append(df)
        all_info.append(info)
    env.close()

    rms_list  = [i['rms_accel']      for i in all_info]
    peak_list = [i['peak_accel']     for i in all_info]
    susp_list = [i['suspension_rms'] for i in all_info]

    print(f'\nEvaluation  algo={args.algo}  road={args.road}  episodes={args.episodes}')
    print(f'  RMS accel:      {np.mean(rms_list):.4f} ± {np.std(rms_list):.4f} m/s²')
    print(f'  Peak accel:     {np.mean(peak_list):.4f} ± {np.std(peak_list):.4f} m/s²')
    print(f'  Suspension RMS: {np.mean(susp_list):.4f} ± {np.std(susp_list):.4f} m\n')

    if args.save:
        import os
        os.makedirs('results', exist_ok=True)
        for i, df in enumerate(all_dfs):
            out = f'results/{args.algo}_{args.road}_ep{i}.csv'
            df.to_csv(out, index=False)
            print(f'  Saved {out}')

    if args.plot:
        df_passive = None
        if args.algo != 'passive':
            penv = QuarterCarEnv(road_profile=args.road)
            df_passive, _ = _run_episode(penv, None, 'passive')
            penv.close()
        _plot(all_dfs[0], df_passive, args.algo)


if __name__ == '__main__':
    main()

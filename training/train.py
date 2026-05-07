"""
Training script for quarter-car active suspension RL.
No ROS imports anywhere in this file.

Usage:
  python training/train.py --algo sac --timesteps 500000
  python training/train.py --algo ppo --road iso_8608_class_c --seed 0
  python training/train.py --algo sac --resume models/sac_xyz/checkpoints/sac_50000_steps.zip
"""
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback
from stable_baselines3.common.monitor import Monitor

from quarter_car_core.quarter_car_env import QuarterCarEnv
from quarter_car_core.wrappers import ActionRepeat, NormalizeObservation, RewardScaler, EpisodeLogger
from quarter_car_core.reward import RewardConfig

ALGOS = {'sac': SAC, 'td3': TD3, 'ppo': PPO}

ALGO_DEFAULTS = {
    'sac': dict(
        policy='MlpPolicy', learning_rate=3e-4, buffer_size=200_000,
        batch_size=256, tau=0.005, gamma=0.99, ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
    'td3': dict(
        policy='MlpPolicy', learning_rate=1e-3, buffer_size=200_000,
        batch_size=256, tau=0.005, gamma=0.99,
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    'ppo': dict(
        policy='MlpPolicy', learning_rate=3e-4, n_steps=2048,
        batch_size=64, n_epochs=10, gamma=0.99,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    ),
}


class ComfortCallback(BaseCallback):
    """Log comfort metrics to TensorBoard; compare vs passive baseline."""

    def __init__(self, make_env_fn, eval_freq: int = 10_000,
                 n_eval_eps: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self._make_env_fn = make_env_fn
        self._eval_freq   = eval_freq
        self._n_eval_eps  = n_eval_eps

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq == 0:
            self._run_comfort_eval()
        return True

    def _run_comfort_eval(self):
        env = self._make_env_fn()
        buckets = {'rms_accel': [], 'peak_accel': [],
                   'comfort_score': [], 'suspension_rms': []}
        for _ in range(self._n_eval_eps):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            for k in buckets:
                buckets[k].append(info.get(k, 0.0))
        env.close()
        for k, v in buckets.items():
            self.logger.record(f'eval/{k}', float(np.mean(v)))


def _make_train_env(road_profile, seed, log_dir, reward_scale=0.1, n_repeat=2):
    def _init():
        env = QuarterCarEnv(road_profile=road_profile)
        env = ActionRepeat(env, n_repeat=n_repeat)
        env = NormalizeObservation(env)
        env = RewardScaler(env, scale=reward_scale)
        env = EpisodeLogger(env, log_dir=log_dir)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def _passive_baseline(road_profile: str, n_eps: int = 5) -> dict:
    env = QuarterCarEnv(road_profile=road_profile)
    metrics = {'rms_accel': [], 'peak_accel': [], 'suspension_rms': []}
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, t, tr, info = env.step(np.array([0.0]))
            done = t or tr
    for k in metrics:
        metrics[k].append(info.get(k, 0.0))
    env.close()
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def _print_comparison(passive: dict, trained: dict):
    print('\n' + '=' * 62)
    print(f"{'Metric':<22} {'Passive':>10} {'Trained':>10} {'Improv%':>10}")
    print('-' * 62)
    for k in ['rms_accel', 'peak_accel', 'suspension_rms']:
        p, t = passive.get(k, 0.0), trained.get(k, 0.0)
        imp = 100.0 * (p - t) / max(abs(p), 1e-9)
        print(f"{k:<22} {p:>10.4f} {t:>10.4f} {imp:>9.1f}%")
    print('=' * 62 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for quarter-car suspension')
    parser.add_argument('--algo',      choices=['sac', 'td3', 'ppo'], default='sac')
    parser.add_argument('--timesteps', type=int,  default=500_000)
    parser.add_argument('--road',      default='iso_8608_class_c',
                        choices=['speed_bump', 'iso_8608_class_c', 'sine_sweep', 'flat'])
    parser.add_argument('--eval-road', default='speed_bump',
                        choices=['speed_bump', 'sine_sweep'])
    parser.add_argument('--config',    default=None, help='Path to env_config.yaml')
    parser.add_argument('--seed',      type=int,  default=42)
    parser.add_argument('--resume',    default=None, help='Path to checkpoint .zip')
    args = parser.parse_args()

    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path(f'models/{args.algo}_{ts}')
    log_dir   = Path(f'logs/{args.algo}_{ts}')
    (model_dir / 'best').mkdir(parents=True, exist_ok=True)
    (model_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_envs    = 4 if args.algo in ('sac', 'td3') else 1
    VecClass  = SubprocVecEnv if args.algo in ('sac', 'td3') else DummyVecEnv

    env_fns = [
        _make_train_env(args.road, args.seed + i, str(log_dir))
        for i in range(n_envs)
    ]
    venv = VecClass(env_fns)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)

    def _eval_env_factory():
        return QuarterCarEnv(road_profile=args.eval_road)

    algo_kwargs = ALGO_DEFAULTS[args.algo].copy()
    if args.resume:
        model = ALGOS[args.algo].load(args.resume, env=venv)
    else:
        model = ALGOS[args.algo](
            env=venv, verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            **algo_kwargs,
        )

    callbacks = [
        SB3EvalCallback(
            _eval_env_factory(),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            best_model_save_path=str(model_dir / 'best'),
            log_path=str(model_dir),
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(model_dir / 'checkpoints'),
            name_prefix=args.algo,
        ),
        ComfortCallback(_eval_env_factory, eval_freq=10_000, n_eval_eps=5),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    model.save(str(model_dir / f'{args.algo}_final'))
    venv.save(str(model_dir / 'vecnormalize.pkl'))
    print(f'\nModel saved to {model_dir}')

    # Passive vs trained comparison table
    passive = _passive_baseline(args.eval_road)
    eval_env = QuarterCarEnv(road_profile=args.eval_road)
    t_buckets = {'rms_accel': [], 'peak_accel': [], 'suspension_rms': []}
    for _ in range(5):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(a)
            done = term or trunc
        for k in t_buckets:
            t_buckets[k].append(info.get(k, 0.0))
    eval_env.close()
    trained = {k: float(np.mean(v)) for k, v in t_buckets.items()}
    _print_comparison(passive, trained)


if __name__ == '__main__':
    main()

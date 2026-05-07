"""
Optuna hyperparameter search for quarter-car suspension RL.
No ROS imports.

Usage:
  python training/hyperparameter_search.py --algo sac --trials 50 --timesteps 50000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).parent.parent / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from quarter_car_core.quarter_car_env import QuarterCarEnv
from quarter_car_core.wrappers import ActionRepeat, RewardScaler

_ALGO_CLS = {'sac': SAC, 'td3': TD3, 'ppo': PPO}


def _make_env(road: str = 'iso_8608_class_c', seed: int = 0):
    env = QuarterCarEnv(road_profile=road)
    env = ActionRepeat(env, n_repeat=2)
    env = RewardScaler(env, scale=0.1)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def _sample_sac_params(trial: optuna.Trial) -> dict:
    n_units = trial.suggest_categorical('n_units', [128, 256, 512])
    return {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'buffer_size':   trial.suggest_categorical('buffer', [100_000, 200_000, 500_000]),
        'batch_size':    trial.suggest_categorical('batch',  [128, 256, 512]),
        'tau':           trial.suggest_float('tau', 0.001, 0.05),
        'gamma':         trial.suggest_float('gamma', 0.95, 0.999),
        'ent_coef':      'auto',
        'policy_kwargs': dict(net_arch=[n_units, n_units]),
    }


def _sample_td3_params(trial: optuna.Trial) -> dict:
    n_units = trial.suggest_categorical('n_units', [256, 400])
    return {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'buffer_size':   trial.suggest_categorical('buffer', [100_000, 200_000]),
        'batch_size':    trial.suggest_categorical('batch',  [128, 256]),
        'tau':           trial.suggest_float('tau', 0.001, 0.05),
        'gamma':         trial.suggest_float('gamma', 0.95, 0.999),
        'policy_kwargs': dict(net_arch=[n_units, n_units]),
    }


def _sample_ppo_params(trial: optuna.Trial) -> dict:
    n_units = trial.suggest_categorical('n_units', [128, 256])
    return {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'n_steps':       trial.suggest_categorical('n_steps', [1024, 2048]),
        'batch_size':    trial.suggest_categorical('batch',   [64, 128]),
        'n_epochs':      trial.suggest_int('n_epochs', 5, 20),
        'gamma':         trial.suggest_float('gamma', 0.95, 0.999),
        'policy_kwargs': dict(net_arch=dict(pi=[n_units, n_units],
                                            vf=[n_units, n_units])),
    }


_SAMPLERS = {'sac': _sample_sac_params, 'td3': _sample_td3_params, 'ppo': _sample_ppo_params}


def _objective(trial: optuna.Trial, algo: str, n_timesteps: int) -> float:
    params = _SAMPLERS[algo](trial)
    venv   = DummyVecEnv([lambda: _make_env(seed=trial.number)])
    model  = _ALGO_CLS[algo]('MlpPolicy', venv, verbose=0,
                              seed=trial.number, **params)
    model.learn(total_timesteps=n_timesteps)
    venv.close()

    run_env = _make_env(road='speed_bump', seed=999)
    episode_rewards = []
    for _ in range(3):
        obs, _ = run_env.reset()
        ep_r, done = 0.0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, t, tr, _ = run_env.step(a)
            ep_r += r
            done  = t or tr
        episode_rewards.append(ep_r)
    run_env.close()
    return float(np.mean(episode_rewards))


def main():
    parser = argparse.ArgumentParser(description='Optuna HP search for quarter-car RL')
    parser.add_argument('--algo',       default='sac', choices=['sac', 'td3', 'ppo'])
    parser.add_argument('--trials',     type=int, default=50)
    parser.add_argument('--timesteps',  type=int, default=50_000)
    parser.add_argument('--study-name', default=None)
    parser.add_argument('--storage',    default=None, help='Optuna storage URL')
    args = parser.parse_args()

    study_name = args.study_name or f'{args.algo}_quarter_car'
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda t: _objective(t, args.algo, args.timesteps),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    print('\nBest hyperparameters:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    print(f'Best episode return: {study.best_value:.4f}')


if __name__ == '__main__':
    main()

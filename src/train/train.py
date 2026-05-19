import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback
from stable_baselines3.common.monitor import Monitor

from QuarterCar_env.envs import QuarterCarEnv
from QuarterCar_env.wrappers import ActionRepeat, NormalizeObservation, RewardScaler, EpisodeLogger

ALGOS = {'sac': SAC, 'td3': TD3, 'ppo': PPO}


def load_configs(config_path=None):
    base = (Path(config_path).parent if config_path
            else Path(__file__).parent / 'configs')
    with open(base / 'env_config.yaml') as f:
        env_cfg = yaml.safe_load(f)
    with open(base / 'algo_configs.yaml') as f:
        algo_cfg = yaml.safe_load(f)
    return env_cfg, algo_cfg


class ComfortCallback(BaseCallback):
    """Log comfort metrics to TensorBoard each eval cycle."""

    def __init__(self, make_env_fn, eval_freq=10_000, n_eval_eps=5, verbose=0):
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


def build_vec_env(algo, road, seed, log_dir, env_cfg, gamma):
    train_cfg     = env_cfg['training']
    reward_scale  = train_cfg['reward_scale']
    action_repeat = train_cfg['action_repeat']
    n_envs        = train_cfg['n_envs'] if algo in ('sac', 'td3') else 1
    VecClass      = SubprocVecEnv if algo in ('sac', 'td3') else DummyVecEnv
    # DummyVecEnv for PPO

    def _single_env(i):
        def _init():
            env = QuarterCarEnv(road_profile=road)
            env = ActionRepeat(env, n_repeat=action_repeat)
            env = NormalizeObservation(env)
            env = RewardScaler(env, scale=reward_scale)
            env = EpisodeLogger(env, log_dir=log_dir)
            env = Monitor(env)
            env.reset(seed=seed + i)
            return env
        return _init

    venv = VecClass([_single_env(i) for i in range(n_envs)])
    return VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)


def build_model(algo, venv, algo_kwargs, log_dir, seed, resume=None):
    if resume:
        return ALGOS[algo].load(resume, env=venv)
    return ALGOS[algo](
        env=venv,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed,
        **algo_kwargs,
    )


def build_callbacks(algo, eval_road, model_dir, training_venv):
    def _raw_eval_env():
        return QuarterCarEnv(road_profile=eval_road)

    # Must match training env wrapper type so SB3 can sync normalization stats.
    eval_venv = VecNormalize(
        DummyVecEnv([_raw_eval_env]),
        norm_obs=False,
        norm_reward=False,
        gamma=training_venv.gamma,
    )

    return [
        SB3EvalCallback(
            eval_venv,
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            best_model_save_path=str(model_dir / 'best'),
            log_path=str(model_dir),
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(model_dir / 'checkpoints'),
            name_prefix=algo,
        ),
        ComfortCallback(_raw_eval_env, eval_freq=10_000, n_eval_eps=5),
    ]


def _passive_baseline(road_profile, n_eps=5):
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


def compare_passive_vs_trained(model, eval_road, n_eps=5):
    passive = _passive_baseline(eval_road, n_eps)

    env     = QuarterCarEnv(road_profile=eval_road)
    buckets = {'rms_accel': [], 'peak_accel': [], 'suspension_rms': []}
    for _ in range(n_eps):
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        for k in buckets:
            buckets[k].append(info.get(k, 0.0))
    env.close()
    trained = {k: float(np.mean(v)) for k, v in buckets.items()}

    print('\n' + '=' * 62)
    print(f"{'Metric':<22} {'Passive':>10} {'Trained':>10} {'Improv%':>10}")
    print('-' * 62)
    for k in ['rms_accel', 'peak_accel', 'suspension_rms']:
        p, t = passive.get(k, 0.0), trained.get(k, 0.0)
        imp  = 100.0 * (p - t) / max(abs(p), 1e-9)
        print(f"{k:<22} {p:>10.4f} {t:>10.4f} {imp:>9.1f}%")
    print('=' * 62 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Train an RL agent for active suspension control on the Quarter Car Model'
    )
    parser.add_argument(
        '--algo',
        type=str,
        default='sac',
        choices=['sac', 'td3', 'ppo'],
        help='RL algorithm to train.',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500_000,
        help='Total environment steps to train for.',
    )
    parser.add_argument(
        '--road',
        type=str,
        default='iso_8608_class_c',
        choices=['speed_bump', 'iso_8608_class_c', 'sine_sweep', 'flat'],
        help='Road profile used during training.',
    )
    parser.add_argument(
        '--eval-road',
        type=str,
        default='speed_bump',
        choices=['speed_bump', 'sine_sweep'],
        help='Road profile used for evaluation callbacks.',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to env_config.yaml. Defaults to training/configs/env_config.yaml.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=66,
        help='Random seed.',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a checkpoint .zip to resume training from.',
    )
    args = parser.parse_args()

    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path(f'models/{args.algo}_{ts}')
    log_dir   = Path(f'logs/{args.algo}_{ts}')
    (model_dir / 'best').mkdir(parents=True, exist_ok=True)
    (model_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, algo_cfg = load_configs(args.config)
    algo_kwargs = algo_cfg[args.algo].copy()
    gamma       = algo_kwargs.get('gamma', 0.99)

    venv  = build_vec_env(args.algo, args.road, args.seed, str(log_dir), env_cfg, gamma)
    model = build_model(args.algo, venv, algo_kwargs, log_dir, args.seed, args.resume)
    cbs   = build_callbacks(args.algo, args.eval_road, model_dir, venv)

    model.learn(total_timesteps=args.timesteps, callback=cbs, progress_bar=True)
    model.save(str(model_dir / f'{args.algo}_final'))
    venv.save(str(model_dir / 'vecnormalize.pkl'))
    print(f'\nModel saved to {model_dir}')

    compare_passive_vs_trained(model, args.eval_road)


if __name__ == '__main__':
    main()

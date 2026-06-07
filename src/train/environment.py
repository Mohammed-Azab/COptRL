from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.wrappers import EpisodeLogger, PreviewWrapper

ENV_ID = "QuarterCar_env/QuarterCar"


def _make_env(
    road: str,
    seed: int,
    monitor_dir: str | None,
    env_kwargs: dict,
) -> Callable[[], gym.Env]:

    def _init() -> gym.Env:
        env = gym.make(ENV_ID, road_profile=road, **env_kwargs)
        env = PreviewWrapper(env)
        env = Monitor(env, monitor_dir)
        if monitor_dir:
            env = EpisodeLogger(env, log_dir=monitor_dir)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init


def make_vec_env(
    road: str,
    n_envs: int,
    base_seed: int,
    monitor_dir: str | None = None,
    gamma: float = 0.99,
    norm_obs: bool = True,
    norm_reward: bool = True,
    env_kwargs: dict | None = None,
) -> VecNormalize:
    """
    Build a DummyVecEnv wrapped in VecNormalize.

    Each worker gets seed = base_seed + worker_index so trajectories are
    diverse but fully reproducible from base_seed.
    """
    env_kwargs = env_kwargs or {}
    fns = [
        _make_env(road, base_seed + i, monitor_dir, env_kwargs)
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(fns)
    venv = VecNormalize(
        venv,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        gamma=gamma,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    return venv


def make_eval_vec_env(
    road: str,
    n_envs: int,
    base_seed: int,
    train_venv: VecNormalize,
    monitor_dir: str | None = None,
    env_kwargs: dict | None = None,
) -> VecNormalize:
    """
    Build an evaluation VecNormalize that shares normalisation stats with the
    training env (obs running mean/var, no reward normalisation).
    """
    env_kwargs = env_kwargs or {}
    fns = [
        _make_env(road, base_seed + i, monitor_dir, env_kwargs)
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(fns)
    eval_venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        gamma=train_venv.gamma,
        clip_obs=train_venv.clip_obs,
    )
    eval_venv.obs_rms  = train_venv.obs_rms
    eval_venv.ret_rms  = train_venv.ret_rms
    eval_venv.training = False
    return eval_venv

"""
Optuna objective: build a model from sampled params, train briefly, evaluate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
from stable_baselines3 import PPO, SAC, TD3

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "gym_env"))
sys.path.insert(0, str(_ROOT / "src"))       # exposes road.road_generator
sys.path.insert(0, str(_ROOT / "src" / "train"))

from env_factory import make_eval_vec_env, make_vec_env
from samplers import sample

_REGISTRY = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
_OFF_POLICY = {"SAC", "TD3"}


class Objective:
    """
    Callable Optuna objective.

    Keeps all state (algo, roads, timestep budget, seed) in the instance so
    the trial function is a clean `__call__(trial) -> float`.
    """

    def __init__(
        self,
        algo: str,
        train_road: str,
        eval_road: str,
        timesteps: int,
        n_eval_episodes: int,
        seed: int,
    ):
        self.algo            = algo.upper()
        self.train_road      = train_road
        self.eval_road       = eval_road
        self.timesteps       = timesteps
        self.n_eval_episodes = n_eval_episodes
        self.seed            = seed

    def __call__(self, trial: optuna.Trial) -> float:
        params = sample(self.algo, trial)
        trial_seed = self.seed + trial.number

        # short-lived envs: no reward normalisation (too short to converge stats)
        train_venv = make_vec_env(
            road=self.train_road,
            n_envs=1,
            base_seed=trial_seed,
            norm_obs=True,
            norm_reward=(self.algo not in _OFF_POLICY),
            gamma=params.get("gamma", 0.99),
        )

        model = _REGISTRY[self.algo](
            "MlpPolicy",
            train_venv,
            verbose=0,
            seed=trial_seed,
            **params,
        )
        model.learn(total_timesteps=self.timesteps)
        train_venv.close()

        # evaluation on a fresh env with frozen normalisation stats
        eval_venv = make_eval_vec_env(
            road=self.eval_road,
            n_envs=1,
            base_seed=trial_seed + 5_000,
            train_venv=model.get_vec_normalize_env(),
        )

        returns = []
        for _ in range(self.n_eval_episodes):
            obs, _ = eval_venv.reset()
            ep_return, done = 0.0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_venv.step(action)
                ep_return += float(reward[0])
                done = bool(terminated[0] or truncated[0])
            returns.append(ep_return)
        eval_venv.close()

        return float(np.mean(returns))

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
import yaml
from stable_baselines3 import PPO

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "gym_env"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "train"))

from environment import make_eval_vec_env, make_vec_env
from search_spaces import sample_from_config

_TUNE_CONFIG_PATH = _ROOT / "config" / "algo" / "tune_config.yaml"


def _load_space(config_path: Path = _TUNE_CONFIG_PATH) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh).get("PPO", {})


class Objective:
    """Optuna objective: sample PPO hyperparameters, train briefly, return eval return."""

    def __init__(
        self,
        train_road: str,
        eval_road: str,
        timesteps: int,
        n_eval_episodes: int,
        seed: int,
        use_curriculum: bool = True,
        config_path: Path = _TUNE_CONFIG_PATH,
    ):
        self.train_road      = train_road
        self.eval_road       = eval_road
        self.timesteps       = timesteps
        self.n_eval_episodes = n_eval_episodes
        self.seed            = seed
        self.use_curriculum  = use_curriculum
        self._space          = _load_space(config_path)

    def __call__(self, trial: optuna.Trial) -> float:
        params     = sample_from_config(trial, self._space)
        trial_seed = self.seed + trial.number

        curriculum_cfg = None
        if self.use_curriculum:
            from QuarterCar_env.wrappers.curriculum import load_curriculum_config
            curriculum_cfg = load_curriculum_config(
                _ROOT / "config" / "curriculum" / "curriculum_params.yaml"
            )

        train_venv = make_vec_env(
            road=self.train_road,
            n_envs=1,
            base_seed=trial_seed,
            norm_obs=True,
            norm_reward=True,
            gamma=params.get("gamma", 0.99),
            curriculum_cfg=curriculum_cfg,
        )

        model = PPO(
            "MlpPolicy",
            train_venv,
            verbose=0,
            seed=trial_seed,
            **params,
        )
        model.learn(total_timesteps=self.timesteps)
        train_venv.close()

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

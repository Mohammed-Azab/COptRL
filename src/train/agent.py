# Algorithm registry and model builder.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
    "TD3": TD3,
}

_OFF_POLICY = {"TD3"}


def supported_algos() -> list[str]:
    return list(_REGISTRY.keys())


def load_algo_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def build_model(
    algo: str,
    venv: VecEnv,
    algo_kwargs: dict,
    tensorboard_log: str,
    seed: int,
    resume: str | None = None,
) -> BaseAlgorithm:
    cls    = _REGISTRY[algo]
    kw     = copy.deepcopy(algo_kwargs)
    policy = kw.pop("policy", "MlpPolicy")

    if resume:
        model = cls.load(
            resume,
            env=venv,
            seed=seed,
            tensorboard_log=tensorboard_log,
            **kw,
        )
    else:
        model = cls(
            policy,
            venv,
            verbose=1,
            seed=seed,
            tensorboard_log=tensorboard_log,
            **kw,
        )
    return model


def is_off_policy(algo: str) -> bool:
    return algo.upper() in _OFF_POLICY

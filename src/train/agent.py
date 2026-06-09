# Algorithm registry and model builder.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
}

_OFF_POLICY: set[str] = set()


def supported_algos() -> list[str]:
    return list(_REGISTRY.keys())


def load_algo_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _parse_lr(lr_val):
    # "lin_X" → linear decay from X to 0  (SB3 convention: callable(progress_remaining))
    if isinstance(lr_val, str) and lr_val.startswith("lin_"):
        initial = float(lr_val[4:])
        return lambda p: initial * p
    return lr_val


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
    if "learning_rate" in kw:
        kw["learning_rate"] = _parse_lr(kw["learning_rate"])

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

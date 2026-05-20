"""
Optuna hyperparameter samplers — one function per algorithm.

Each function receives a `trial` and returns a dict of SB3 constructor kwargs
(ready to be passed directly to the model).
"""

from __future__ import annotations

import optuna


def sample_ppo(trial: optuna.Trial) -> dict:
    n_units = trial.suggest_categorical("n_units", [128, 256, 512])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs":      trial.suggest_int("n_epochs", 5, 20),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.90, 0.99),
        "policy_kwargs": {"net_arch": {"pi": [n_units, n_units], "vf": [n_units, n_units]}},
    }


def sample_sac(trial: optuna.Trial) -> dict:
    n_units = trial.suggest_categorical("n_units", [128, 256, 512])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":   trial.suggest_categorical("buffer_size", [100_000, 200_000, 500_000]),
        "batch_size":    trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau":           trial.suggest_float("tau", 0.001, 0.05),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "ent_coef":      "auto",
        "policy_kwargs": {"net_arch": [n_units, n_units]},
    }


def sample_td3(trial: optuna.Trial) -> dict:
    n_units = trial.suggest_categorical("n_units", [256, 400])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size":   trial.suggest_categorical("buffer_size", [100_000, 200_000]),
        "batch_size":    trial.suggest_categorical("batch_size", [128, 256]),
        "tau":           trial.suggest_float("tau", 0.001, 0.05),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "policy_kwargs": {"net_arch": [n_units, n_units]},
    }


_SAMPLERS: dict[str, callable] = {
    "PPO": sample_ppo,
    "SAC": sample_sac,
    "TD3": sample_td3,
}


def sample(algo: str, trial: optuna.Trial) -> dict:
    """Dispatch to the right sampler by algorithm name (case-insensitive)."""
    key = algo.upper()
    if key not in _SAMPLERS:
        raise ValueError(f"No sampler for algo '{algo}'. Available: {list(_SAMPLERS)}")
    return _SAMPLERS[key](trial)

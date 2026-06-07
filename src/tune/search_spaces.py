from __future__ import annotations

import optuna


def sample_from_config(trial: optuna.Trial, space: dict) -> dict:
    """Build SB3 kwargs from a YAML search-space dict.

    Supported types: float_log, float, int, categorical.
    n_units is a special key: resolved to policy_kwargs.net_arch.
    """
    params: dict = {}
    n_units = None

    for name, spec in space.items():
        kind = spec["type"]

        if name == "n_units":
            n_units = trial.suggest_categorical("n_units", spec["choices"])
            continue

        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search-space type '{kind}' for param '{name}'")

    if n_units is not None:
        params["policy_kwargs"] = {
            "net_arch": {"pi": [n_units, n_units], "vf": [n_units, n_units]}
        }

    return params

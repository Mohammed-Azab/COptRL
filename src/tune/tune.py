from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "gym_env"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "train"))
sys.path.insert(0, str(_ROOT / "src" / "tune"))

from trial import Objective

_TUNE_CONFIG_PATH = _ROOT / "config" / "algo" / "tune_config.yaml"
_RESULTS_DIR      = _ROOT / "tune" / "results"
_VALID_ROADS      = ["speed_bump", "flat", "recorded"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna PPO hyperparameter search for COptRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trials",       type=int, default=50)
    p.add_argument("--timesteps",    type=int, default=None,
                   help="Env steps per trial. Defaults to tune_config.yaml.")
    p.add_argument("--train-road",   default=None, choices=_VALID_ROADS)
    p.add_argument("--eval-road",    default=None, choices=_VALID_ROADS)
    p.add_argument("--seed",         type=int, default=None)
    p.add_argument("--n-jobs",       type=int, default=1,
                   help="Parallel Optuna workers (requires --storage).")
    p.add_argument("--study-name",   default=None,
                   help="Reuse an existing study by name.")
    p.add_argument("--storage",      default=None,
                   help="Optuna storage URL, e.g. sqlite:///tune.db")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Disable curriculum wrapper during tuning trials.")
    p.add_argument("--config",       default=str(_TUNE_CONFIG_PATH))
    return p.parse_args()


def _load_defaults(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("defaults", {})


def _trial_to_dict(trial: optuna.trial.FrozenTrial) -> dict:
    return {
        "number":   trial.number,
        "value":    trial.value,
        "state":    trial.state.name,
        "params":   trial.params,
        "datetime": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
    }


def save_results(study: optuna.Study, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    payload = {
        "study_name":         study.study_name,
        "algo":               "PPO",
        "direction":          study.direction.name,
        "n_trials_completed": len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "best_trial": {
            "number": best.number,
            "value":  best.value,
            "params": best.params,
        },
        "all_trials": [_trial_to_dict(t) for t in study.trials],
    }
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nResults saved → {output_path}")


def main() -> None:
    args     = parse_args()
    defaults = _load_defaults(args.config)

    timesteps       = args.timesteps  or defaults.get("timesteps_per_trial", 100_000)
    train_road      = args.train_road or defaults.get("train_road", "speed_bump")
    eval_road       = args.eval_road  or defaults.get("eval_road",  "speed_bump")
    seed            = args.seed       or defaults.get("seed", 0)
    n_eval_ep       = defaults.get("n_eval_episodes", 5)
    use_curriculum  = not args.no_curriculum and defaults.get("use_curriculum", True)

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name   = args.study_name or f"PPO_{ts}"
    results_path = _RESULTS_DIR / f"{study_name}.json"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    objective = Objective(
        train_road=train_road,
        eval_road=eval_road,
        timesteps=timesteps,
        n_eval_episodes=n_eval_ep,
        seed=seed,
        use_curriculum=use_curriculum,
        config_path=Path(args.config),
    )

    print(f"\n  COptRL — PPO Hyperparameter Search")
    print(f"  study       : {study_name}")
    print(f"  trials      : {args.trials}")
    print(f"  steps/trial : {timesteps:,}")
    print(f"  train road  : {train_road}  |  eval : {eval_road}")
    print(f"  curriculum  : {'on' if use_curriculum else 'off'}")
    print(f"  seed        : {seed}\n")

    best_so_far: float | None = None

    def _on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_so_far
        if trial.value is None:
            return
        completed = len([t for t in study.trials if t.value is not None])
        is_best   = best_so_far is None or trial.value > best_so_far
        if is_best:
            best_so_far = trial.value
            print(f"  trial {completed:>3d}/{args.trials}  return={trial.value:+.3f}  ★ new best"
                  f"  lr={trial.params.get('learning_rate', '?'):.1e}"
                  f"  n_steps={trial.params.get('n_steps', '?')}"
                  f"  clip={trial.params.get('clip_range', '?')}")
        elif completed % 5 == 0:
            print(f"  trial {completed:>3d}/{args.trials}  return={trial.value:+.3f}"
                  f"  best={best_so_far:+.3f}")

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
            callbacks=[_on_trial_end],
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving results so far...")
    finally:
        completed = [t for t in study.trials if t.value is not None]
        if not completed:
            print("No trials completed — nothing to save.")
            return
        print("\n  Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"    {k:20s}: {v}")
        print(f"    {'value':20s}: {study.best_value:.4f}")
        save_results(study, results_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import yaml
from tqdm import tqdm as _tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "gym_env"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "train"))
sys.path.insert(0, str(_ROOT / "src" / "tune"))

from trial import Objective

_TUNE_CONFIG_PATH  = _ROOT / "config" / "algo" / "tune_config.yaml"
_ALGO_CONFIG_PATH  = _ROOT / "config" / "algo" / "algo_configs.yaml"
_TUNE_ROOT         = _ROOT / "tune"
_VALID_ROADS       = ["speed_bump", "flat", "recorded"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna PPO hyperparameter search for COptRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trials",        type=int, default=50)
    p.add_argument("--timesteps",     type=int, default=None,
                   help="Env steps per trial. Defaults to tune_config.yaml.")
    p.add_argument("--train-road",    default=None, choices=_VALID_ROADS)
    p.add_argument("--eval-road",     default=None, choices=_VALID_ROADS)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--n-jobs",        type=int, default=1,
                   help="Parallel Optuna workers (requires --storage).")
    p.add_argument("--study-name",    default=None,
                   help="Reuse an existing study by name.")
    p.add_argument("--storage",       default=None,
                   help="Optuna storage URL, e.g. sqlite:///tune.db")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Disable curriculum wrapper during tuning trials.")
    p.add_argument("--config",        default=str(_TUNE_CONFIG_PATH))
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


def save_results(study: optuna.Study, out_dir: Path) -> None:
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
    out_path = out_dir / "results.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"Results (JSON)     → {out_path}")


class _InlineLists(yaml.Dumper):
    # render Python lists as inline YAML sequences: [256, 256] not block style
    pass

_InlineLists.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def save_best_yaml(study: optuna.Study, out_dir: Path) -> None:
    with open(_ALGO_CONFIG_PATH) as fh:
        base = yaml.safe_load(fh).get("PPO", {})

    best = dict(study.best_params)
    n_units = int(best.pop("n_units", 256))

    config = {**base, **best}
    config["policy_kwargs"] = {
        "net_arch": {"pi": [n_units, n_units], "vf": [n_units, n_units]}
    }

    out_path = out_dir / "best_params.yaml"
    with open(out_path, "w") as fh:
        yaml.dump({"PPO": config}, fh, Dumper=_InlineLists,
                  default_flow_style=False, sort_keys=False)
    print(f"Best params (YAML) → {out_path}")


def main() -> None:
    args     = parse_args()
    defaults = _load_defaults(args.config)

    timesteps      = args.timesteps  if args.timesteps  is not None else defaults.get("timesteps_per_trial", 100_000)
    train_road     = args.train_road or defaults.get("train_road", "speed_bump")
    eval_road      = args.eval_road  or defaults.get("eval_road",  "speed_bump")
    seed           = args.seed       if args.seed       is not None else defaults.get("seed", 0)
    n_eval_ep      = defaults.get("n_eval_episodes", 5)
    use_curriculum = not args.no_curriculum and defaults.get("use_curriculum", True)

    study_name = args.study_name or "myPPO_study"

    # auto-increment exp_n inside tune/<study_name>/<train_road>/
    study_root = _TUNE_ROOT / study_name / train_road
    study_root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_")[1]) for p in study_root.iterdir()
                if p.is_dir() and p.name.startswith("exp_") and p.name.split("_")[1].isdigit()]
    exp_id  = (max(existing) + 1) if existing else 1
    out_dir = study_root / f"exp_{exp_id}"
    out_dir.mkdir()

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
    print(f"  output      : {out_dir}")
    print(f"  trials      : {args.trials}")
    print(f"  steps/trial : {timesteps:,}")
    print(f"  train road  : {train_road}  |  eval : {eval_road}")
    print(f"  curriculum  : {'on' if use_curriculum else 'off'}")
    print(f"  seed        : {seed}\n")

    best_so_far: float | None = None
    trial_count: int = 0

    _tqdm.write(
        f"  {'#':>3}  {'return':>8}  {'time':>5}  {'lr':>9}  {'steps':>5}  "
        f"{'batch':>5}  {'ent':>9}  {'clip':>4}  {'n_units':>7}  {'epochs':>6}  note"
    )
    _tqdm.write(
        f"  {'-'*3}  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*5}  "
        f"{'-'*5}  {'-'*9}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*14}"
    )

    def _on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_so_far, trial_count
        trial_count += 1
        dur_s = (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if trial.datetime_complete and trial.datetime_start else 0
        )
        if trial.value is None:
            _tqdm.write(f"  {trial_count:>3}/{args.trials}  {'FAILED':>8}  {dur_s:>4.0f}s")
            return

        p       = trial.params
        is_best = best_so_far is None or trial.value > best_so_far
        if is_best:
            best_so_far = trial.value
            note = "★ new best"
        else:
            note = f"best={best_so_far:+.3f}"

        _tqdm.write(
            f"  {trial_count:>3}/{args.trials}  {trial.value:>+8.3f}  {dur_s:>4.0f}s"
            f"  {p.get('learning_rate', 0):>9.2e}"
            f"  {p.get('n_steps', 0):>5d}"
            f"  {p.get('batch_size', 0):>5d}"
            f"  {p.get('ent_coef', 0):>9.2e}"
            f"  {p.get('clip_range', 0):.2f}"
            f"  {p.get('n_units', 0):>7d}"
            f"  {p.get('n_epochs', 0):>6d}"
            f"  {note}"
        )

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
        print(f"    {'value':20s}: {study.best_value:.4f}\n")
        save_results(study, out_dir)
        save_best_yaml(study, out_dir)


if __name__ == "__main__":
    main()

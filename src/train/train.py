from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# resolve src tree 
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "gym_env"))
sys.path.insert(0, str(_ROOT / "src")) 
sys.path.insert(0, str(_ROOT / "src" / "train"))

from agent import build_model, load_algo_config, supported_algos
from monitoring import build_callbacks
from environment import make_eval_vec_env, make_vec_env
from seed import seed_everything
from QuarterCar_env.config.reward_params import load_reward_config
from QuarterCar_env.config.env_params import EPISODE_STEPS
from QuarterCar_env.reward.utils import reward_bounds
from QuarterCar_env.wrappers.curriculum import load_curriculum_config


_CONFIG_PATH      = _ROOT / "config" / "algo" / "algo_configs.yaml"
_CURRICULUM_PATH  = _ROOT / "config" / "curriculum" / "curriculum_params.yaml"
_VALID_ROADS = ["speed_bump", "flat", "recorded"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a speed-planning RL agent on the quarter-car environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
                    "--algo",
                    default="PPO",
                    choices=supported_algos())
    p.add_argument(
                    "--road",
                    default="speed_bump",
                    choices=_VALID_ROADS)
    p.add_argument(
                    "--eval-road",
                    default="speed_bump",
                    choices=_VALID_ROADS,
                    help="Road for eval callbacks")
    p.add_argument(
                    "--seed",
                    type=int,
                    default=69,
                    help="Random seed.")
    p.add_argument(
                    "--timesteps",
                    type=int,
                    default=None,
                    help="Total env steps.")
    p.add_argument(
                    "--n-envs",
                    type=int, 
                    default=None,
                    help="Parallel training envs.")
    p.add_argument(
                    "--no-normalize",
                    action="store_true",
                    help="Disable VecNormalize.")
    p.add_argument(
                    "--resume",
                    default=None,
                    help="Checkpoint .zip to continue training from.")
    p.add_argument(
                    "--run-name",
                    default=None,
                    help="Tag appended to the output directory name.")
    p.add_argument(
                    "--config",
                    default=str(_CONFIG_PATH),
                    help="Path to algo_configs.yaml.")
    p.add_argument(
                    "--render",
                    action="store_true",
                    default=False,
                    help="Enable rendering during training (requires a display; slows training).")
    p.add_argument(
                    "--curriculum",
                    action="store_true",
                    default=False,
                    help="Wrap training env with CurriculumWrapper (speed_bump only).")

    return p.parse_args()


def _summarize_monitor(monitor_dir: Path) -> dict[str, float] | None:
    # SB3 Monitor
    files = sorted(monitor_dir.glob("*.monitor.csv"))
    if not files:
        sibling = monitor_dir.parent / f"{monitor_dir.name}.monitor.csv"
        if sibling.exists():
            files = [sibling]
    if not files:
        return None

    rewards: list[float] = []
    lengths: list[int] = []

    for file_path in files:
        with file_path.open(newline="") as handle:
            reader = csv.DictReader(
                row for row in handle if not row.startswith("#")
            )
            for row in reader:
                if "r" in row and "l" in row:
                    rewards.append(float(row["r"]))
                    lengths.append(int(row["l"]))

    if not rewards:
        return None

    total = float(len(rewards))
    mean_reward = sum(rewards) / total
    mean_length = sum(lengths) / total
    return {
        "episodes": float(len(rewards)),
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "last_reward": rewards[-1],
    }


def _write_summary(
    path: Path,
    args,
    algo_kwargs: dict,
    timesteps: int,
    n_envs: int,
    seed: int,
    curriculum: bool,
    final_path: Path,
    best_path: Path,
    best_step: int | None,
    monitor_summary: dict | None,
    interrupted: bool,
    error: str | None,
) -> None:
    flags: list[str] = []
    if monitor_summary:
        mr = monitor_summary["mean_reward"]
        mn = monitor_summary["min_reward"]
        mx = monitor_summary["max_reward"]
        ep = int(monitor_summary["episodes"])
        if mr < 0:
            flags.append(f"mean_reward={mr:.1f} is negative — reward shaping may need review")
        if mx - mr > abs(mr) * 3:
            flags.append(f"high reward variance (max={mx:.1f}, mean={mr:.1f}) — unstable training")
        if mn < -500:
            flags.append(f"extreme negative episode (min={mn:.1f}) — possible truncation or reward bug")
        if ep < 100:
            flags.append(f"only {ep} episodes completed — likely insufficient timesteps or early truncation")

    if error:
        next_steps = ["Read the error_traceback field and fix the crash before retraining"]
    elif interrupted:
        next_steps = [f"Resume: just train --resume {final_path}.zip"]
    else:
        next_steps = [
            f"Evaluate: just eval {final_path}.zip --save-plots",
            f"Compare:  just compare {final_path}.zip",
            "Check TRIAL_ERROR.md if mean_reward looks wrong",
            "View curves: just tb",
        ]

    payload = {
        "date":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status":     "INTERRUPTED" if interrupted else "ERROR" if error else "COMPLETED",
        "config": {
            "algo":       args.algo,
            "road":       args.road,
            "curriculum": curriculum,
            "timesteps":  timesteps,
            "n_envs":     n_envs,
            "seed":       seed,
            "norm_obs":   True,
            "norm_reward": not args.no_normalize,
            "resume":     args.resume,
        },
        "hyperparameters": algo_kwargs,
        "paths": {
            "final_model": str(final_path) + ".zip",
            "best_model":  str(best_path)  + ".zip",
            "best_step":   best_step,
            "vecnorm":     str(final_path.parent / "vecnormalize.pkl"),
        },
        "training_stats": monitor_summary,
        "diagnostics": flags if flags else ["none — training stats look normal"],
        "next_steps":  next_steps,
        "error_traceback": error,
    }

    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _best_model_step(best_dir: Path) -> int | None:
    eval_file = best_dir / "evaluations.npz"
    if not eval_file.exists():
        return None

    with np.load(eval_file) as data:
        if "timesteps" not in data or "results" not in data:
            return None

        timesteps = np.asarray(data["timesteps"]).reshape(-1)
        results = np.asarray(data["results"])

    if timesteps.size == 0 or results.size == 0:
        return None

    mean_rewards = results.mean(axis=1)
    if mean_rewards.size == 0:
        return None

    best_so_far = np.maximum.accumulate(mean_rewards)
    improved = np.r_[True, best_so_far[1:] > best_so_far[:-1]]
    last_best_idx = int(np.where(improved)[0][-1])
    return int(timesteps[last_best_idx])


def main() -> None:
    args = parse_args()

    full_cfg   = load_algo_config(args.config)
    train_meta = full_cfg["training"]
    algo_kwargs = dict(full_cfg[args.algo])   # shallow copy; build_model pops 'policy'

    # CLI overrides win over config defaults
    seed      = args.seed      if args.seed      is not None else train_meta["seed"]
    timesteps = args.timesteps if args.timesteps is not None else train_meta["total_timesteps"]
    n_envs    = args.n_envs    if args.n_envs    is not None else train_meta["n_envs"]
    eval_road = args.eval_road or train_meta["eval_road"]
    normalize = not args.no_normalize

    seed_everything(seed)

    rcfg           = load_reward_config()
    curriculum_cfg = load_curriculum_config(_CURRICULUM_PATH) if args.curriculum else None

    #  output directories
    exp_root = _ROOT / "models" / args.algo / args.road
    if args.run_name is None:
        exp_re = re.compile(r"^exp_(\d+)$")
        exp_ids = []
        if exp_root.exists():
            for path in exp_root.iterdir():
                if path.is_dir():
                    match = exp_re.match(path.name)
                    if match:
                        exp_ids.append(int(match.group(1)))
        next_id = (max(exp_ids) + 1) if exp_ids else 1
        run_tag = f"exp_{next_id}"
    else:
        run_tag = args.run_name

    model_dir = exp_root / run_tag
    tb_dir    = _ROOT / "logs" / "tensorboard" / run_tag
    mon_dir   = _ROOT / "logs" / "monitor" / run_tag
    best_path = model_dir / "best" / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    mon_dir.mkdir(parents=True, exist_ok=True)

    #  environments
    gamma = algo_kwargs.get("gamma", 0.99)
    norm_reward = normalize and train_meta.get("norm_reward", True)

    render_mode = "human" if args.render else "none"
    train_venv = make_vec_env(
        road=args.road,
        n_envs=n_envs,
        base_seed=seed,
        monitor_dir=str(mon_dir / "train"),
        gamma=gamma,
        norm_obs=normalize,
        norm_reward=norm_reward,
        env_kwargs={"render_mode": render_mode},
        curriculum_cfg=curriculum_cfg,
    )
    eval_venv = make_eval_vec_env(
        road=eval_road,
        n_envs=train_meta["n_eval_envs"],
        base_seed=seed + 10_000,   # disjoint seed range keeps eval trajectories fresh
        train_venv=train_venv,
        monitor_dir=str(mon_dir / "eval"),
    )

    model = build_model(
        algo=args.algo,
        venv=train_venv,
        algo_kwargs=algo_kwargs,
        tensorboard_log=str(tb_dir),
        seed=seed,
        resume=args.resume,
    )

    callbacks = build_callbacks(
        model_dir=model_dir,
        eval_venv=eval_venv,
        train_venv=train_venv,
        eval_freq=max(train_meta["eval_freq"] // n_envs, 1),
        n_eval_episodes=train_meta["n_eval_episodes"],
        checkpoint_freq=max(train_meta["checkpoint_freq"] // n_envs, 1),
    )

    bounds = reward_bounds(rcfg, EPISODE_STEPS)

    print(f"\n{''*58}")
    print(f"  algo       : {args.algo}")
    print(f"  road       : {args.road}  |  eval : {eval_road}")
    print(f"  seed       : {seed}")
    print(f"  timesteps  : {timesteps:,}")
    print(f"  n_envs     : {n_envs}")
    print(f"  normalize  : obs={normalize}, reward={norm_reward}")
    print(f"  render     : {args.render}")
    print(f"  v_max      : {rcfg.v_max * 3.6:.0f} km/h  |  v_min : {rcfg.v_min * 3.6:.1f} km/h")
    print(f"  preview    : {rcfg.n_peaks} peaks × 3 = {rcfg.n_peaks * 3} features over {rcfg.preview_distance}m")
    print(f"  curriculum : {'on (' + str(len(curriculum_cfg['thresholds'])) + ' levels)' if curriculum_cfg else 'off'}")
    print(f"  output     : {model_dir}")
    print(f"  reward range  episode  [{bounds['episode_min']:+.0f},  {bounds['episode_max']:+.0f}]")
    print(f"                per-step [{bounds['per_step_min']:+.2f}, {bounds['per_step_max']:+.2f}]")
    print(f"{''*58}\n")

    _interrupted = False
    _error: str | None = None
    final_path = model_dir / f"{args.algo}_final"

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )
    except KeyboardInterrupt:
        _interrupted = True
        print("\nInterrupted — saving checkpoint...")
    except Exception:
        _error = traceback.format_exc()
        print(f"\nTraining crashed:\n{_error}")
    finally:
        model.save(str(final_path))
        train_venv.save(str(model_dir / "vecnormalize.pkl"))
        train_venv.close()
        eval_venv.close()

    print(f"\nModel    → {final_path}.zip")
    print(f"VecNorm  → {model_dir / 'vecnormalize.pkl'}")
    print(f"TB logs  → tensorboard --logdir {tb_dir}")

    best_step    = _best_model_step(model_dir / "best")
    mon_summary  = _summarize_monitor(mon_dir / "train")

    print(f"\nBest Model     → {best_path}.zip")
    print(f"Best Step      → {f'{best_step:,}' if best_step is not None else 'unknown'}")

    if mon_summary:
        emin, emax = bounds['episode_min'], bounds['episode_max']
        mean_r = mon_summary['mean_reward']
        max_r  = mon_summary['max_reward']
        min_r  = mon_summary['min_reward']
        erange = emax - emin if emax != emin else 1.0

        print("\nTraining summary")
        print(f"  episodes     : {int(mon_summary['episodes'])}")
        print(f"  mean_reward  : {mean_r:.1f}   ({100*(mean_r-emin)/erange:.0f}% of range)")
        print(f"  max_reward   : {max_r:.1f}   ({100*(max_r-emin)/erange:.0f}% of range)")
        print(f"  min_reward   : {min_r:.1f}   ({100*(min_r-emin)/erange:.0f}% of range)")
        print(f"  last_reward  : {mon_summary['last_reward']:.1f}")
        print(f"  mean_ep_len  : {mon_summary['mean_length']:.1f}")
        print(f"\n  reward range  episode  [{emin:+.0f}, {emax:+.0f}]")
        print(f"                per-step [{bounds['per_step_min']:+.2f}, {bounds['per_step_max']:+.2f}]")
    else:
        print("\nTraining summary: no monitor data found.")

    summary_path = model_dir / "summary.json"
    _write_summary(
        path=summary_path,
        args=args,
        algo_kwargs=algo_kwargs,
        timesteps=timesteps,
        n_envs=n_envs,
        seed=seed,
        curriculum=args.curriculum,
        final_path=final_path,
        best_path=best_path,
        best_step=best_step,
        monitor_summary=mon_summary,
        interrupted=_interrupted,
        error=_error,
    )
    print(f"Summary  → {summary_path}")


if __name__ == "__main__":
    main()

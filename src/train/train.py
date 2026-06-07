from __future__ import annotations

import argparse
import csv
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
    lines: list[str] = []

    def s(text: str = "") -> None:
        lines.append(text)

    s("# Training Run Summary")
    s(f"date        : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    s(f"status      : {'INTERRUPTED' if interrupted else ('ERROR — ' + error) if error else 'COMPLETED'}")
    s()

    s("## Run Configuration")
    s(f"algo        : {args.algo}")
    s(f"road        : {args.road}")
    s(f"curriculum  : {'yes' if curriculum else 'no'}")
    s(f"timesteps   : {timesteps:,}")
    s(f"n_envs      : {n_envs}")
    s(f"seed        : {seed}")
    s(f"normalize   : obs=True, reward={not args.no_normalize}")
    s(f"resume      : {args.resume or 'no'}")
    s()

    s("## Hyperparameters (PPO)")
    for k, v in algo_kwargs.items():
        if k == "policy_kwargs":
            arch = v.get("net_arch", {})
            s(f"  net_arch    : pi={arch.get('pi', '?')}  vf={arch.get('vf', '?')}")
        else:
            s(f"  {k:<20}: {v}")
    s()

    s("## Output Paths")
    s(f"final model : {final_path}.zip")
    s(f"best model  : {best_path}.zip")
    s(f"best step   : {f'{best_step:,}' if best_step is not None else 'unknown'}")
    s(f"vecnorm     : {final_path.parent / 'vecnormalize.pkl'}")
    s()

    s("## Training Statistics")
    if monitor_summary:
        s(f"episodes    : {int(monitor_summary['episodes'])}")
        s(f"mean_reward : {monitor_summary['mean_reward']:.3f}")
        s(f"max_reward  : {monitor_summary['max_reward']:.3f}")
        s(f"min_reward  : {monitor_summary['min_reward']:.3f}")
        s(f"last_reward : {monitor_summary['last_reward']:.3f}")
        s(f"mean_ep_len : {monitor_summary['mean_length']:.1f}")
        s()
        # flags for Claude to catch obvious failure modes
        flags: list[str] = []
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
        if flags:
            s("## Warnings / Diagnostics")
            for f in flags:
                s(f"  ! {f}")
            s()
        else:
            s("## Warnings / Diagnostics")
            s("  none — training stats look normal")
            s()
    else:
        s("  no monitor data found — training may not have completed any episodes")
        s()

    if error:
        s("## Error Traceback")
        s(error)
        s()

    s("## What To Check Next")
    if error:
        s("  - Read the traceback above and fix the crash before retraining")
    elif interrupted:
        s("  - Resume from checkpoint: just train --resume <final_path>.zip")
    else:
        s("  - Evaluate: just eval <final_path>.zip --save-plots")
        s("  - Compare vs baselines: just compare <final_path>.zip")
        s("  - If mean_reward is low, check TRIAL_ERROR.md for known failure patterns")
        s("  - View training curves: just tb")

    path.write_text("\n".join(lines) + "\n")


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

    print(f"\n{''*58}")
    print(f"  algo       : {args.algo}")
    print(f"  road       : {args.road}  |  eval : {eval_road}")
    print(f"  seed       : {seed}")
    print(f"  timesteps  : {timesteps:,}")
    print(f"  n_envs     : {n_envs}")
    print(f"  normalize  : obs={normalize}, reward={norm_reward}")
    print(f"  render     : {args.render}")
    print(f"  preview    : {rcfg.n_peaks} peaks × 3 = {rcfg.n_peaks * 3} features over {rcfg.preview_distance}m")
    print(f"  curriculum : {'on (' + str(len(curriculum_cfg['thresholds'])) + ' levels)' if curriculum_cfg else 'off'}")
    print(f"  output     : {model_dir}")
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
        print("\nTraining summary")
        print(f"  episodes     : {int(mon_summary['episodes'])}")
        print(f"  mean_reward  : {mon_summary['mean_reward']:.3f}")
        print(f"  mean_ep_len  : {mon_summary['mean_length']:.1f}")
        print(f"  max_reward   : {mon_summary['max_reward']:.3f}")
        print(f"  min_reward   : {mon_summary['min_reward']:.3f}")
        print(f"  last_reward  : {mon_summary['last_reward']:.3f}")
    else:
        print("\nTraining summary: no monitor data found.")

    summary_path = model_dir / "summary.md"
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

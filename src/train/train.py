"""
Quarter-car RL training entry point.

Usage
-----
  python src/train/train.py --algo PPO
  python src/train/train.py --algo SAC --road iso_8608_class_c --seed 7
  python src/train/train.py --algo PPO --resume models/PPO_20240101/checkpoints/ckpt_50000_steps.zip

All hyperparameters live in config/algo/algo_configs.yaml.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ── resolve src tree so this script runs from the project root ────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "gym_env"))
sys.path.insert(0, str(_ROOT / "src"))       # exposes road.road_generator
sys.path.insert(0, str(_ROOT / "src" / "train"))

from algo_factory import build_model, is_off_policy, load_algo_config, supported_algos
from callbacks import build_callbacks
from env_factory import make_eval_vec_env, make_vec_env
from seed_utils import seed_everything


_CONFIG_PATH = _ROOT / "config" / "algo" / "algo_configs.yaml"
_VALID_ROADS = ["iso_8608_class_c", "speed_bump", "sine_sweep", "flat"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a speed-planning RL agent on the quarter-car environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo",         default="PPO",  choices=supported_algos())
    p.add_argument("--road",         default="iso_8608_class_c", choices=_VALID_ROADS)
    p.add_argument("--eval-road",    default=None,   choices=_VALID_ROADS,
                   help="Road for eval callbacks. Defaults to training road.")
    p.add_argument("--seed",         type=int, default=None,
                   help="Random seed. Overrides algo_configs.yaml.")
    p.add_argument("--timesteps",    type=int, default=None,
                   help="Total env steps. Overrides algo_configs.yaml.")
    p.add_argument("--n-envs",       type=int, default=None,
                   help="Parallel training envs.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable VecNormalize.")
    p.add_argument("--resume",       default=None,
                   help="Checkpoint .zip to continue training from.")
    p.add_argument("--run-name",     default=None,
                   help="Tag appended to the output directory name.")
    p.add_argument("--config",       default=str(_CONFIG_PATH),
                   help="Path to algo_configs.yaml.")
    return p.parse_args()


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

    # ── output directories ────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.algo}_{ts}" + (f"_{args.run_name}" if args.run_name else "")

    model_dir = _ROOT / "models" / run_tag
    tb_dir    = _ROOT / "logs" / "tensorboard" / run_tag
    mon_dir   = _ROOT / "logs" / "monitor" / run_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    mon_dir.mkdir(parents=True, exist_ok=True)

    # ── environments ─────────────────────────────────────────────────────────
    gamma = algo_kwargs.get("gamma", 0.99)
    # off-policy (SAC/TD3): normalising rewards distorts Q-value targets
    norm_reward = normalize and train_meta.get("norm_reward_ppo", True) and not is_off_policy(args.algo)

    train_venv = make_vec_env(
        road=args.road,
        n_envs=n_envs,
        base_seed=seed,
        monitor_dir=str(mon_dir / "train"),
        gamma=gamma,
        norm_obs=normalize,
        norm_reward=norm_reward,
    )
    eval_venv = make_eval_vec_env(
        road=eval_road,
        n_envs=train_meta["n_eval_envs"],
        base_seed=seed + 10_000,   # disjoint seed range keeps eval trajectories fresh
        train_venv=train_venv,
        monitor_dir=str(mon_dir / "eval"),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        algo=args.algo,
        venv=train_venv,
        algo_kwargs=algo_kwargs,
        tensorboard_log=str(tb_dir),
        seed=seed,
        resume=args.resume,
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = build_callbacks(
        model_dir=model_dir,
        eval_venv=eval_venv,
        train_venv=train_venv,
        eval_freq=max(train_meta["eval_freq"] // n_envs, 1),
        n_eval_episodes=train_meta["n_eval_episodes"],
        checkpoint_freq=max(train_meta["checkpoint_freq"] // n_envs, 1),
    )

    # ── run ───────────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  algo       : {args.algo}")
    print(f"  road       : {args.road}  |  eval : {eval_road}")
    print(f"  seed       : {seed}")
    print(f"  timesteps  : {timesteps:,}")
    print(f"  n_envs     : {n_envs}")
    print(f"  normalize  : obs={normalize}, reward={norm_reward}")
    print(f"  output     : {model_dir}")
    print(f"{'─'*58}\n")

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.resume is None),
    )

    # ── save ──────────────────────────────────────────────────────────────────
    final_path = model_dir / f"{args.algo}_final"
    model.save(str(final_path))
    train_venv.save(str(model_dir / "vecnormalize.pkl"))

    print(f"\nModel    → {final_path}.zip")
    print(f"VecNorm  → {model_dir / 'vecnormalize.pkl'}")
    print(f"TB logs  → tensorboard --logdir {tb_dir}")

    train_venv.close()
    eval_venv.close()


if __name__ == "__main__":
    main()

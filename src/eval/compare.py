from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("src/gym_env", "src", "src/train"):
    sys.path.insert(0, str(_ROOT / _p))

import gymnasium as gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.wrappers import PreviewWrapper
from QuarterCar_env.reward.utils import reward_bounds
from QuarterCar_env.config.reward_params import load_reward_config
from QuarterCar_env.config.env_params import EPISODE_STEPS

_ALGO_MAP: dict[str, type] = {"PPO": PPO, "TD3": TD3}
_VALID_ROADS = ["speed_bump", "flat", "recorded"]
_DEFAULT_CFG = _ROOT / "config" / "eval" / "compare_config.yaml"
_ENV_ID = "QuarterCar_env/QuarterCar"

_SCALAR_KEYS = [
    "total_return",
    "mean_step_reward",
    "rms_accel",
    "peak_accel",
    "speed_rmse",
    "comfort_score",
    "action_smoothness_rms",
]

# CLI + config

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare RL agent vs baselines across road profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to YAML config (default: config/eval/compare_config.yaml).")
    p.add_argument("--algo", choices=[k for k in list(_ALGO_MAP) + [k.lower() for k in _ALGO_MAP]],
                   default=None)
    p.add_argument("--model-path", default=None,
                   help="Path to trained model .zip.")
    p.add_argument("--vecnorm-path", default=None,
                   help="Path to vecnormalize .pkl. Auto-inferred when omitted.")
    p.add_argument("--road", choices=_VALID_ROADS + ["all"], default=None,
                   help="Road profile(s). 'all' evaluates all three.")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="Episodes per (agent × road) pair.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--render", action="store_true",
                   help="Enable simulation rendering (requires a display).")
    p.add_argument("--save-plots", action="store_true",
                   help="Save matplotlib figures to results_dir.")
    p.add_argument("--results-dir", default=None,
                   help="Output directory for JSON + figures.")
    p.add_argument("--no-baselines", action="store_true",
                   help="Skip baseline agents; evaluate trained agent only.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.config) if args.config else _DEFAULT_CFG
    with open(cfg_path) as f:
        cfg: dict = yaml.safe_load(f)

    # CLI overrides (None = not provided → keep YAML value)
    if args.algo:                   cfg["algo"]        = args.algo.upper()
    if args.model_path:             cfg["model_path"]  = args.model_path
    if args.vecnorm_path:           cfg["vecnorm_path"] = args.vecnorm_path
    if args.road:
        cfg["roads"] = _VALID_ROADS if args.road == "all" else [args.road]
    if args.n_episodes is not None: cfg["n_episodes"]  = args.n_episodes
    if args.seed is not None:       cfg["seed"]        = args.seed
    if args.render:                 cfg["render"]      = True
    if args.save_plots:             cfg["save_plots"]  = True
    if args.results_dir:            cfg["results_dir"] = args.results_dir
    if args.no_baselines:           cfg["baselines"]   = []

    # Validate required fields
    if not cfg.get("model_path"):
        raise ValueError("model_path is required. Pass --model-path or set it in the config YAML.")
    if cfg.get("algo", "").upper() not in _ALGO_MAP:
        raise ValueError(f"algo must be one of {list(_ALGO_MAP)}. Got: {cfg.get('algo')}")

    return cfg

# Agent loading

def _infer_vecnorm(model_path: Path) -> Path:
    candidate = model_path.parent / "vecnormalize.pkl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find vecnormalize.pkl near {model_path}.\n"
        "Pass --vecnorm-path or set vecnorm_path in the config."
    )


def load_agent(cfg: dict):
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vecnorm_path = (
        Path(cfg["vecnorm_path"]) if cfg.get("vecnorm_path")
        else _infer_vecnorm(model_path)
    )
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    cls = _ALGO_MAP[cfg["algo"].upper()]
    model = cls.load(str(model_path))
    return model, vecnorm_path

# Episode rollout helpers

def _record_step(ep: dict, action: float, reward: float, info: dict) -> None:
    ep["rewards"].append(reward)
    ep["actions"].append(action)
    ep["speeds"].append(info.get("speed", 0.0))
    ep["v_refs"].append(info.get("v_ref", 0.0))
    ep["body_accels"].append(info.get("z_B_ddot", 0.0))
    ep["comfort_scores"].append(info.get("comfort_score", 0.0))
    ep["rms_accel_running"].append(info.get("rms_accel", 0.0))
    for key in ("r_heave", "r_wheel", "r_tracking", "r_accel", "r_jerk", "r_action_smooth", "r_curve"):
        ep[key].append(info.get(key, 0.0))


def _summarize(ep: dict) -> dict:
    rewards = np.array(ep["rewards"])
    accels  = np.array(ep["body_accels"])
    speeds  = np.array(ep["speeds"])
    v_refs  = np.array(ep["v_refs"])
    actions = np.array(ep["actions"])

    return {
        "total_return":          float(rewards.sum()),
        "mean_step_reward":      float(rewards.mean()),
        "n_steps":               int(len(rewards)),
        "rms_accel":             float(np.sqrt(np.mean(accels ** 2))),
        "peak_accel":            float(np.max(np.abs(accels))),
        "speed_rmse":            float(np.sqrt(np.mean((speeds - v_refs) ** 2))),
        "comfort_score":         float(ep["comfort_scores"][-1]) if ep["comfort_scores"] else 0.0,
        "action_smoothness_rms": float(np.sqrt(np.mean(np.diff(actions) ** 2))) if len(actions) > 1 else 0.0,
        "ts": {k: [float(x) for x in v] for k, v in ep.items()},
    }


def _reset_vec(venv: VecNormalize):
    # SB3 >= 2.0 returns (obs, info); older returns just obs
    result = venv.reset()
    return result[0] if isinstance(result, tuple) else result


def rollout_trained(
    model,
    vecnorm_path: Path,
    road: str,
    render: bool,
    deterministic: bool,
) -> dict:
    render_mode = "human" if render else "none"

    def _env_fn():
        env = gym.make(_ENV_ID, road_profile=road, render_mode=render_mode,
                       random_road_on_reset=False)
        return PreviewWrapper(env)

    venv = DummyVecEnv([_env_fn])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False

    obs = _reset_vec(venv)
    done_arr = np.array([False])
    ep: dict = defaultdict(list)

    while not done_arr[0]:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_result = venv.step(action)
        # Handle both SB3 4-tuple and 5-tuple step APIs
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info_list = step_result
            done_arr = np.logical_or(terminated, truncated)
        else:
            obs, reward, done_arr, info_list = step_result
        _record_step(ep, float(action[0, 0]), float(reward[0]), info_list[0])

    venv.close()
    return _summarize(ep)


def rollout_baseline(road: str, policy: str, seed: int, render: bool) -> dict:
    render_mode = "human" if render else "none"
    env = gym.make(_ENV_ID, road_profile=road, render_mode=render_mode,
                   random_road_on_reset=False)
    env = PreviewWrapper(env)
    obs, _ = env.reset(seed=seed)
    done = False
    ep: dict = defaultdict(list)

    while not done:
        if policy == "passive":
            action = np.array([0.0], dtype=np.float32)
        elif policy == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(f"Unknown baseline policy: {policy!r}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        _record_step(ep, float(action[0]), float(reward), info)

    env.close()
    return _summarize(ep)

# Multi-episode runner + aggregation

def run_agent(
    agent_name: str,
    road: str,
    cfg: dict,
    model=None,
    vecnorm_path: Path | None = None,
) -> list[dict]:
    n = cfg["n_episodes"]
    seed = cfg["seed"]
    render = cfg.get("render", False)
    deterministic = cfg.get("deterministic", True)
    episodes = []

    for ep_i in range(n):
        if agent_name == "trained":
            ep = rollout_trained(model, vecnorm_path, road, render, deterministic)
        else:
            ep = rollout_baseline(road, agent_name, seed + ep_i, render)

        episodes.append(ep)
        print(
            f"  [{agent_name:8s} | {road:20s} | ep {ep_i + 1:>3d}/{n}]"
            f"  return={ep['total_return']:+9.1f}"
            f"  rms_accel={ep['rms_accel']:.3f} m/s²"
            f"  comfort={ep['comfort_score']:.3f}"
        )

    return episodes


def aggregate(episodes: list[dict]) -> dict:
    out: dict = {"n_episodes": len(episodes)}
    for key in _SCALAR_KEYS:
        vals = [ep[key] for ep in episodes]
        out[key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "all":  [float(v) for v in vals],
        }
    # Keep first episode's time-series as a representative for plotting
    out["representative_ts"] = episodes[0]["ts"]
    return out

# Console output

def _table(title: str, metric: str, unit: str, results: dict) -> None:
    agents = list(results.keys())
    roads  = list(next(iter(results.values())).keys())
    W, C = 22, 22
    sep = "─" * (W + C * len(agents) + 4)

    print(f"\n  {title} {unit}")
    print(f"  {sep}")
    header = f"  {'road':<{W}}" + "".join(f"{'mean ± std':>{C}}" for _ in agents)
    sub    = f"  {' ':<{W}}" + "".join(f"{a:>{C}}" for a in agents)
    print(header)
    print(sub)
    print(f"  {sep}")

    for road in roads:
        row = f"  {road:<{W}}"
        for agent in agents:
            agg = results[agent][road]
            m = agg[metric]["mean"]
            s = agg[metric]["std"]
            row += f"  {m:+9.2f} ±{s:5.2f}    "
        print(row)
    print(f"  {sep}")


def print_summary(results: dict, bounds: dict, cfg: dict) -> None:
    agents = list(results.keys())
    roads  = list(next(iter(results.values())).keys())
    print(f"\n{'═' * 72}")
    print(f"  QUARTER-CAR COMPARISON  |  algo={cfg['algo']}"
          f"  n_episodes={cfg['n_episodes']}  roads={len(roads)}")
    print(f"  reward bounds / episode:"
          f"  min={bounds['episode_min']:.1f}   max={bounds['episode_max']:.1f}"
          f"   (per step: [{bounds['per_step_min']:.2f}, {bounds['per_step_max']:.1f}])")
    print(f"  agents evaluated: {', '.join(agents)}")
    print(f"{'═' * 72}")

    _table("EPISODE RETURN",          "total_return",  "",       results)
    _table("RMS BODY ACCEL [m/s²]",   "rms_accel",     "",       results)
    _table("SPEED TRACKING RMSE [m/s]","speed_rmse",   "",       results)
    _table("COMFORT SCORE [0–1]",     "comfort_score", "",       results)
    print()

# Plotting

_AGENT_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def _agent_color(i: int) -> str:
    return _AGENT_COLORS[i % len(_AGENT_COLORS)]


def _bar_group(ax, roads, agents, data_fn, ylabel, title, ref_lines=None):
    import matplotlib.pyplot as plt  # noqa: F401

    n_agents = len(agents)
    x = np.arange(len(roads))
    width = 0.75 / max(n_agents, 1)

    for i, agent in enumerate(agents):
        means = [data_fn(agent, road, "mean") for road in roads]
        stds  = [data_fn(agent, road, "std")  for road in roads]
        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width * 0.92,
            label=agent, color=_agent_color(i),
            yerr=stds, capsize=4, alpha=0.85, error_kw={"lw": 1.2},
        )

    if ref_lines:
        for value, label, color, ls in ref_lines:
            ax.axhline(value, color=color, ls=ls, lw=1.4, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in roads], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_returns(results: dict, bounds: dict, save_dir: Path | None) -> None:
    import matplotlib.pyplot as plt

    agents = list(results.keys())
    roads  = list(next(iter(results.values())).keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Episode Return — Trained Agent vs Baselines", fontsize=13, fontweight="bold")

    # Left: raw return
    ref_raw = [
        (bounds["episode_max"], f"theoretical max ({bounds['episode_max']:.0f})", "green",  "--"),
        (bounds["episode_min"], f"theoretical min ({bounds['episode_min']:.0f})", "crimson","--"),
        (0,                     "zero return",                                    "gray",   ":"),
    ]
    _bar_group(
        axes[0], roads, agents,
        lambda ag, rd, stat: results[ag][rd]["total_return"][stat],
        "episode return", "Raw Episode Return", ref_raw,
    )

    # Right: normalised performance [0–100 %]
    ep_range = bounds["episode_max"] - bounds["episode_min"]

    def norm_mean(ag, rd, stat):
        raw = results[ag][rd]["total_return"][stat]
        if stat == "mean":
            return 100.0 * (raw - bounds["episode_min"]) / ep_range
        # std in normalised space
        return 100.0 * raw / ep_range

    ref_norm = [
        (100, "100 % (theoretical best)", "green",  "--"),
        (0,   "0 % (theoretical worst)", "crimson", "--"),
    ]
    _bar_group(
        axes[1], roads, agents, norm_mean,
        "% of optimal", "Normalised Performance [0–100 %]", ref_norm,
    )

    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rms_accel(results: dict, rcfg, save_dir: Path | None) -> None:
    import matplotlib.pyplot as plt

    agents = list(results.keys())
    roads  = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("RMS Body Acceleration — ISO 2631 Reference", fontsize=13, fontweight="bold")

    ref_accel = [
        (rcfg.a_comfort, f"comfort threshold ({rcfg.a_comfort} m/s²)", "#FF9800", "--"),
        (rcfg.a_limit,   f"discomfort limit  ({rcfg.a_limit} m/s²)",  "crimson",  "--"),
    ]
    _bar_group(
        ax, roads, agents,
        lambda ag, rd, stat: results[ag][rd]["rms_accel"][stat],
        "RMS body acceleration [m/s²]", "RMS Body Acceleration", ref_accel,
    )

    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "rms_accel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_return_distributions(results: dict, bounds: dict, save_dir: Path | None) -> None:
    import matplotlib.pyplot as plt

    agents = list(results.keys())
    roads  = list(next(iter(results.values())).keys())
    n_roads = len(roads)

    fig, axes = plt.subplots(1, n_roads, figsize=(4 * n_roads, 5), sharey=True)
    if n_roads == 1:
        axes = [axes]
    fig.suptitle("Episode Return Distributions", fontsize=13, fontweight="bold")

    for ax, road in zip(axes, roads):
        data = [results[ag][road]["total_return"]["all"] for ag in agents]
        bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.55)
        for patch, i in zip(bp["boxes"], range(len(agents))):
            patch.set_facecolor(_agent_color(i))
            patch.set_alpha(0.75)

        ax.axhline(bounds["episode_max"], color="green",  ls="--", lw=1.2, alpha=0.7,
                   label=f"max ({bounds['episode_max']:.0f})")
        ax.axhline(0,                    color="gray",   ls=":",  lw=1.0, alpha=0.7,
                   label="zero")
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_title(road.replace("_", "\n"), fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("episode return", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "return_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(results: dict, bounds: dict, rcfg, save_dir: Path | None) -> None:
    import matplotlib.pyplot as plt

    agents = list(results.keys())
    roads  = list(next(iter(results.values())).keys())

    for road in roads:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        fig.suptitle(f"Representative Episode — {road}", fontsize=12, fontweight="bold")

        ax_accel, ax_speed, ax_action, ax_reward = axes

        for i, agent in enumerate(agents):
            ts = results[agent][road]["representative_ts"]
            n  = len(ts["rewards"])
            t  = np.arange(n) * 0.02  # DT = 0.02 s

            color = _agent_color(i)
            kw = {"color": color, "lw": 1.5, "label": agent, "alpha": 0.85}

            ax_accel.plot(t, ts["body_accels"], **kw)
            ax_speed.plot(t, ts["speeds"],      **kw)
            ax_speed.plot(t, ts["v_refs"],      color=color, lw=1.0, ls=":", alpha=0.5)
            ax_action.plot(t, ts["actions"],    **kw)
            ax_reward.plot(t, ts["rewards"],    **kw)

        # Reference lines
        ax_accel.axhline( rcfg.a_comfort,  color="#FF9800", ls="--", lw=1.2, label=f"+comfort ({rcfg.a_comfort})")
        ax_accel.axhline(-rcfg.a_comfort,  color="#FF9800", ls="--", lw=1.2)
        ax_accel.axhline( rcfg.a_limit,    color="crimson", ls="--", lw=1.0, label=f"+limit ({rcfg.a_limit})")
        ax_accel.axhline(-rcfg.a_limit,    color="crimson", ls="--", lw=1.0)

        ax_reward.axhline(bounds["per_step_max"], color="green",  ls="--", lw=1.2,
                          label=f"step max ({bounds['per_step_max']:.1f})")
        ax_reward.axhline(bounds["per_step_min"], color="crimson", ls="--", lw=1.2,
                          label=f"step min ({bounds['per_step_min']:.2f})")
        ax_reward.axhline(0, color="gray", ls=":", lw=1.0)

        ax_accel.set_ylabel("body accel [m/s²]", fontsize=9)
        ax_speed.set_ylabel("speed [m/s]",        fontsize=9)
        ax_action.set_ylabel("action [-1, 1]",    fontsize=9)
        ax_reward.set_ylabel("step reward",        fontsize=9)
        ax_reward.set_xlabel("time [s]",           fontsize=9)

        ax_action.set_ylim(-1.1, 1.1)

        for ax in axes:
            ax.legend(fontsize=7, loc="upper right", ncol=max(len(agents), 2))
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        if save_dir:
            fname = f"timeseries_{road}.png"
            fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_plots(results: dict, bounds: dict, rcfg, cfg: dict, save_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving figures to {save_dir} ...")

    plot_returns(results, bounds, save_dir)
    plot_rms_accel(results, rcfg, save_dir)
    plot_return_distributions(results, bounds, save_dir)
    plot_timeseries(results, bounds, rcfg, save_dir)

    print(f"  Saved: returns.png, rms_accel.png, return_distributions.png,"
          f" timeseries_<road>.png")

# JSON export

def save_json(results: dict, bounds: dict, cfg: dict, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"compare_{cfg['algo']}_{ts}.json"

    # Omit raw ts arrays from the JSON summary to keep it readable
    def _strip_ts(agent_road_agg: dict) -> dict:
        return {k: v for k, v in agent_road_agg.items() if k != "representative_ts"}

    export = {
        "timestamp": ts,
        "config": {k: v for k, v in cfg.items() if k not in ("model_path", "vecnorm_path")},
        "model_path": str(cfg.get("model_path", "")),
        "reward_bounds": bounds,
        "results": {
            agent: {road: _strip_ts(agg) for road, agg in roads.items()}
            for agent, roads in results.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"  JSON saved → {out_path}")
    return out_path

# Main

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)

    rcfg   = load_reward_config()
    bounds = reward_bounds(rcfg, EPISODE_STEPS)

    roads   = cfg["roads"]
    n_eps   = cfg["n_episodes"]
    results: dict[str, dict[str, dict]] = {}

    # ── trained agent ─────────────────────────────────────────────────────────
    print(f"\n  Loading {cfg['algo']} model from {cfg['model_path']} ...")
    model, vecnorm_path = load_agent(cfg)
    results["trained"] = {}

    for road in roads:
        print(f"\n[trained / {road}]")
        episodes = run_agent("trained", road, cfg, model=model, vecnorm_path=vecnorm_path)
        results["trained"][road] = aggregate(episodes)

    # ── baselines ─────────────────────────────────────────────────────────────
    for baseline in cfg.get("baselines", []):
        results[baseline] = {}
        for road in roads:
            print(f"\n[{baseline} / {road}]")
            episodes = run_agent(baseline, road, cfg)
            results[baseline][road] = aggregate(episodes)

    # ── reporting ─────────────────────────────────────────────────────────────
    print_summary(results, bounds, cfg)

    save_dir = _ROOT / cfg.get("results_dir", "eval/results")
    save_json(results, bounds, cfg, save_dir)

    if cfg.get("save_plots", False):
        save_plots(results, bounds, rcfg, cfg, save_dir)
    else:
        print("  (pass --save-graphs to generate figures)")


if __name__ == "__main__":
    main()

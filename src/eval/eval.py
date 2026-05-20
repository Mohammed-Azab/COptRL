from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("src/gym_env", "src", "src/train"):
    sys.path.insert(0, str(_ROOT / _p))

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import QuarterCar_env.envs  # noqa: F401
from QuarterCar_env.reward.reward import reward_bounds
from QuarterCar_env.config.reward_params import load_reward_config
from QuarterCar_env.config.env_params import EPISODE_STEPS, DT


_ALGO_MAP: dict[str, type] = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
_VALID_ROADS = ["iso_8608_class_c", "speed_bump", "sine_sweep", "flat"]
_ENV_ID = "QuarterCar_env/QuarterCar"

_REWARD_TERM_KEYS = [
    "r_comfort_bonus",
    "r_tracking",
    "r_accel",
    "r_jerk",
    "r_action_smooth",
    "r_curve",
]



# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep single-agent evaluation on the quarter-car environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo",
                   choices=list(_ALGO_MAP), required=True,
                   help="Algorithm used to train the model.")
    p.add_argument("--model_path", required=True,
                   help="Path to trained model .zip file.")
    p.add_argument("--vecnorm-path", default=None,
                   help="Path to vecnormalize.pkl. Auto-inferred from model directory when omitted.")
    p.add_argument("--road",
                   choices=_VALID_ROADS, default="speed_bump",
                   help="Road profile to evaluate on.")
    p.add_argument("--n-episodes", type=int, default=5,
                   help="Number of episodes to run.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed (each episode gets seed + ep_index).")
    p.add_argument("--render", action="store_true",
                   help="Enable simulation rendering (requires a display).")
    p.add_argument("--save-graphs", action="store_true",
                   help="Save matplotlib figures to results_dir.")
    p.add_argument("--results-dir", default=None,
                   help="Output directory for JSON + figures (default: eval/results/eval_<timestamp>).")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample from policy stochastically instead of taking the mode.")
    return p.parse_args()



# Model loading
def _infer_vecnorm(model_path: Path) -> Path:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "vecnormalize.pkl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find vecnormalize.pkl for {model_path}.\n"
        "Pass --vecnorm-path explicitly."
    )


def load_model(algo: str, model_path: str, vecnorm_path: str | None):
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")

    vp = Path(vecnorm_path) if vecnorm_path else _infer_vecnorm(mp)
    if not vp.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vp}")

    cls = _ALGO_MAP[algo.upper()]
    model = cls.load(str(mp))
    return model, vp


def _record(ep: dict, action: float, reward: float, info: dict) -> None:
    ep["rewards"].append(reward)
    ep["actions"].append(action)
    ep["speeds"].append(info.get("speed", 0.0))
    ep["v_refs"].append(info.get("v_ref", 0.0))
    ep["body_accels"].append(info.get("z_B_ddot", 0.0))
    ep["rms_accel_running"].append(info.get("rms_accel", 0.0))
    ep["comfort_score_running"].append(info.get("comfort_score", 0.0))
    ep["preview_max_height"].append(info.get("preview_max_height", 0.0))
    for k in _REWARD_TERM_KEYS:
        ep[k].append(info.get(k, 0.0))


def run_episode(
    model,
    vecnorm_path: Path,
    road: str,
    seed: int,
    render: bool,
    deterministic: bool,
) -> dict:
    render_mode = "human" if render else "none"
    env_fn = lambda r=road, rm=render_mode: gym.make(_ENV_ID, road_profile=r, render_mode=rm)
    venv = DummyVecEnv([env_fn])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training    = False
    venv.norm_reward = False

    reset_out = venv.reset()
    obs: np.ndarray = reset_out[0] if isinstance(reset_out, tuple) else reset_out  # type: ignore[index]

    done = np.array([False])
    ep: dict = defaultdict(list)

    while not done[0]:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_result: tuple = venv.step(action)  # type: ignore[assignment]
        obs    = step_result[0]
        reward = step_result[1]
        if len(step_result) == 5:
            done      = np.logical_or(step_result[2], step_result[3])
            info_list = step_result[4]
        else:
            done      = step_result[2]
            info_list = step_result[3]
        _record(ep, float(action[0, 0]), float(reward[0]), info_list[0])

    venv.close()
    return dict(ep)



# Per-episode metrics
def episode_metrics(ep: dict) -> dict:
    rewards = np.asarray(ep["rewards"])
    accels  = np.asarray(ep["body_accels"])
    speeds  = np.asarray(ep["speeds"])
    v_refs  = np.asarray(ep["v_refs"])
    actions = np.asarray(ep["actions"])

    metrics = {
        "total_return":          float(rewards.sum()),
        "mean_step_reward":      float(rewards.mean()),
        "n_steps":               int(len(rewards)),
        "rms_accel":             float(np.sqrt(np.mean(accels ** 2))),
        "peak_accel":            float(np.max(np.abs(accels))),
        "speed_rmse":            float(np.sqrt(np.mean((speeds - v_refs) ** 2))),
        "mean_speed":            float(speeds.mean()),
        "comfort_score":         float(ep["comfort_score_running"][-1]),
        "action_rms":            float(np.sqrt(np.mean(actions ** 2))),
        "action_smoothness_rms": float(np.sqrt(np.mean(np.diff(actions) ** 2))) if len(actions) > 1 else 0.0,
    }
    for k in _REWARD_TERM_KEYS:
        metrics[f"total_{k}"] = float(np.sum(ep.get(k, [0.0])))

    return metrics


def aggregate_episodes(all_metrics: list[dict]) -> dict:
    keys = list(all_metrics[0].keys())
    agg: dict = {"n_episodes": len(all_metrics)}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }
    return agg



def print_episode_line(ep_i: int, n: int, m: dict) -> None:
    print(
        f"  ep {ep_i+1:>3d}/{n}"
        f"  return={m['total_return']:+9.1f}"
        f"  rms_accel={m['rms_accel']:.3f} m/s²"
        f"  comfort={m['comfort_score']:.3f}"
        f"  speed_rmse={m['speed_rmse']:.2f} m/s"
    )


def print_summary(agg: dict, bounds: dict, algo: str, road: str) -> None:
    W   = 30
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"  algo={algo}  road={road}  episodes={agg['n_episodes']}")
    print(f"  reward bounds: [{bounds['episode_min']:.0f}, {bounds['episode_max']:.0f}]"
          f"   per-step: [{bounds['per_step_min']:.2f}, {bounds['per_step_max']:.1f}]")
    print(f"{'═'*60}")

    rows = [
        ("Episode return",        "total_return",           ""),
        ("Mean step reward",      "mean_step_reward",        ""),
        ("RMS body accel",        "rms_accel",               "m/s²"),
        ("Peak body accel",       "peak_accel",              "m/s²"),
        ("Speed tracking RMSE",   "speed_rmse",              "m/s"),
        ("Mean speed",            "mean_speed",              "m/s"),
        ("Comfort score",         "comfort_score",           "[0-1]"),
        ("Action smoothness RMS", "action_smoothness_rms",   ""),
    ]

    print(f"  {sep}")
    print(f"  {'Metric':<{W}}  {'mean ± std':>14}   {'min':>8}   {'max':>8}  unit")
    print(f"  {sep}")
    for label, key, unit in rows:
        d = agg[key]
        print(f"  {label:<{W}}  {d['mean']:>+9.3f} ±{d['std']:>6.3f}   {d['min']:>+8.3f}   {d['max']:>+8.3f}  {unit}")
    print(f"  {sep}")

    print(f"\n  Per-term reward totals (summed over steps, mean ± std across episodes):")
    print(f"  {sep}")
    for k in _REWARD_TERM_KEYS:
        fk = f"total_{k}"
        if fk in agg:
            d = agg[fk]
            print(f"  {k:<{W}}  {d['mean']:>+9.2f} ±{d['std']:>6.2f}")
    print(f"  {sep}\n")



def plot_timeseries(ep: dict, ep_i: int, rcfg, bounds: dict, save_dir: Path | None) -> None:
    """6-panel time-series for one episode: body accel, speed, running RMS, action, reward, breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t = np.arange(len(ep["rewards"])) * DT

    fig = plt.figure(figsize=(13, 14))
    gs  = gridspec.GridSpec(6, 1, hspace=0.45)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    ax_accel, ax_speed, ax_rms, ax_action, ax_reward, ax_breakdown = axes

    # Body acceleration + comfort / discomfort bands
    ax_accel.plot(t, ep["body_accels"], color="#1565C0", lw=1.4, label="body accel z̈_B")
    ax_accel.axhspan( rcfg.a_comfort,  rcfg.a_limit,   alpha=0.06, color="orange")
    ax_accel.axhspan(-rcfg.a_limit,   -rcfg.a_comfort, alpha=0.06, color="orange")
    ax_accel.axhline( rcfg.a_comfort,  color="#F57C00", ls="--", lw=1.2,
                      label=f"comfort ±{rcfg.a_comfort} m/s²")
    ax_accel.axhline(-rcfg.a_comfort,  color="#F57C00", ls="--", lw=1.2)
    ax_accel.axhline( rcfg.a_limit,    color="#B71C1C", ls="--", lw=1.0,
                      label=f"limit ±{rcfg.a_limit} m/s²")
    ax_accel.axhline(-rcfg.a_limit,    color="#B71C1C", ls="--", lw=1.0)
    ax_accel.set_ylabel("body accel [m/s²]", fontsize=9)
    ax_accel.legend(fontsize=7, loc="upper right", ncol=2)

    # Speed vs reference
    ax_speed.fill_between(t, ep["speeds"], ep["v_refs"],
                          alpha=0.12, color="#1B5E20", label="tracking error")
    ax_speed.plot(t, ep["speeds"], color="#1B5E20", lw=1.4, label="speed v")
    ax_speed.plot(t, ep["v_refs"], color="#A5D6A7", lw=1.0, ls="--", label="v_ref")
    ax_speed.set_ylabel("speed [m/s]", fontsize=9)
    ax_speed.legend(fontsize=7, loc="upper right")

    # Running RMS accel (comfort trend)
    ax_rms.plot(t, ep["rms_accel_running"], color="#6A1B9A", lw=1.4, label="running RMS accel")
    ax_rms.axhline(rcfg.a_comfort, color="#F57C00", ls="--", lw=1.2,
                   label=f"comfort ({rcfg.a_comfort} m/s²)")
    ax_rms.set_ylabel("RMS accel [m/s²]", fontsize=9)
    ax_rms.legend(fontsize=7, loc="upper right")

    # Action
    ax_action.plot(t, ep["actions"], color="#E65100", lw=1.4, label="action u")
    ax_action.axhline(0, color="gray", ls=":", lw=0.8)
    ax_action.set_ylim(-1.15, 1.15)
    ax_action.set_ylabel("action [-1, 1]", fontsize=9)
    ax_action.legend(fontsize=7, loc="upper right")

    # Total step reward
    ax_reward.plot(t, ep["rewards"], color="#212121", lw=1.4, label="step reward")
    ax_reward.axhline(bounds["per_step_max"], color="green",   ls="--", lw=1.2,
                      label=f"step max ({bounds['per_step_max']:.1f})")
    ax_reward.axhline(bounds["per_step_min"], color="crimson", ls="--", lw=1.2,
                      label=f"step min ({bounds['per_step_min']:.2f})")
    ax_reward.axhline(0, color="gray", ls=":", lw=0.8)
    ax_reward.set_ylabel("step reward", fontsize=9)
    ax_reward.legend(fontsize=7, loc="upper right", ncol=2)

    # Per-term reward breakdown
    term_colors = {
        "r_comfort_bonus": "#43A047",
        "r_tracking":      "#1E88E5",
        "r_accel":         "#E53935",
        "r_jerk":          "#FB8C00",
        "r_action_smooth": "#8E24AA",
        "r_curve":         "#00ACC1",
    }
    for k, color in term_colors.items():
        vals = ep.get(k, [0.0] * len(t))
        if any(v != 0.0 for v in vals):
            ax_breakdown.plot(t, vals, color=color, lw=1.2, alpha=0.85, label=k)
    ax_breakdown.axhline(0, color="gray", ls=":", lw=0.8)
    ax_breakdown.set_ylabel("per-term reward", fontsize=9)
    ax_breakdown.set_xlabel("time [s]", fontsize=9)
    ax_breakdown.legend(fontsize=7, loc="lower right", ncol=3)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Episode {ep_i + 1} — Deep Inspection", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_dir:
        fname = save_dir / f"timeseries_ep{ep_i + 1}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    saved → {fname.name}")
    plt.close(fig)


def plot_episode_comparison(all_metrics: list[dict], rcfg, bounds: dict,
                             save_dir: Path | None) -> None:
    """2×2 bar chart comparing key metrics across all evaluated episodes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_metrics)
    ep_labels = [f"ep{i+1}" for i in range(n)]
    x = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))  # type: ignore[attr-defined]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Per-Episode Metrics Comparison", fontsize=12, fontweight="bold")

    def _bar(ax, vals, ylabel, title, ref_lines=None):
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", lw=0.5)
        if ref_lines:
            for val, label, color, ls in ref_lines:
                ax.axhline(val, color=color, ls=ls, lw=1.4, label=label, alpha=0.85)
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(ep_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    _bar(axes[0, 0],
         [m["total_return"]   for m in all_metrics], "episode return",   "Episode Return",
         [(bounds["episode_max"], f"max={bounds['episode_max']:.0f}", "green", "--"),
          (0, "zero", "gray", ":")])
    _bar(axes[0, 1],
         [m["rms_accel"]      for m in all_metrics], "RMS accel [m/s²]", "RMS Body Acceleration",
         [(rcfg.a_comfort, f"comfort ({rcfg.a_comfort})", "#F57C00", "--"),
          (rcfg.a_limit,   f"limit ({rcfg.a_limit})",    "crimson",  "--")])
    _bar(axes[1, 0],
         [m["comfort_score"]  for m in all_metrics], "comfort score [0-1]", "Comfort Score",
         [(1.0, "perfect=1.0", "green", "--")])
    _bar(axes[1, 1],
         [m["speed_rmse"]     for m in all_metrics], "speed RMSE [m/s]",   "Speed Tracking RMSE",
         [(0, "perfect=0", "green", ":")])

    fig.tight_layout()
    if save_dir:
        fname = save_dir / "episode_comparison.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    saved → {fname.name}")
    plt.close(fig)


def plot_reward_breakdown(all_metrics: list[dict], save_dir: Path | None) -> None:
    """Stacked bar showing each term's total contribution per episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_metrics)
    ep_labels = [f"ep{i+1}" for i in range(n)]
    x = np.arange(n)

    terms = [k for k in _REWARD_TERM_KEYS
             if any(m.get(f"total_{k}", 0) != 0 for m in all_metrics)]
    colors = ["#43A047", "#1E88E5", "#E53935", "#FB8C00", "#8E24AA", "#00ACC1"]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    fig.suptitle("Reward Term Breakdown (total per episode)", fontsize=12, fontweight="bold")

    pos_bottom = np.zeros(n)
    neg_bottom = np.zeros(n)

    for i, k in enumerate(terms):
        vals      = np.array([m.get(f"total_{k}", 0.0) for m in all_metrics])
        pos_vals  = np.where(vals >= 0, vals, 0)
        neg_vals  = np.where(vals <  0, vals, 0)
        color     = colors[i % len(colors)]
        ax.bar(x, pos_vals, bottom=pos_bottom, label=k, color=color, alpha=0.82, edgecolor="white")
        ax.bar(x, neg_vals, bottom=neg_bottom,            color=color, alpha=0.82, edgecolor="white")
        pos_bottom += pos_vals
        neg_bottom += neg_vals

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ep_labels, fontsize=9)
    ax.set_ylabel("summed reward contribution", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", ncol=len(terms))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_dir:
        fname = save_dir / "reward_breakdown.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    saved → {fname.name}")
    plt.close(fig)



# JSON export

def save_json(agg: dict, all_metrics: list[dict], algo: str, road: str,
              model_path: str, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"eval_{algo}_{road}_{ts}.json"

    export = {
        "timestamp":   ts,
        "algo":        algo,
        "road":        road,
        "model_path":  str(model_path),
        "aggregate":   agg,
        "per_episode": all_metrics,
    }

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"  JSON saved → {out_path}")



def main() -> None:
    args = parse_args()
    deterministic = not args.no_deterministic

    rcfg   = load_reward_config()
    bounds = reward_bounds(rcfg, EPISODE_STEPS)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = (
        Path(args.results_dir) if args.results_dir
        else _ROOT / "eval" / "results" / f"eval_{args.algo}_{args.road}_{ts}"
    )

    print(f"\n{'═'*60}")
    print(f"  QUARTER-CAR EVAL")
    print(f"  algo={args.algo}  road={args.road}  episodes={args.n_episodes}")
    print(f"  model: {args.model_path}")
    print(f"  deterministic={deterministic}  render={args.render}")
    print(f"{'═'*60}\n")

    model, vecnorm_path = load_model(args.algo, args.model_path, args.vecnorm_path)

    all_eps:     list[dict] = []
    all_metrics: list[dict] = []

    for ep_i in range(args.n_episodes):
        ep = run_episode(
            model, vecnorm_path,
            road=args.road,
            seed=args.seed + ep_i,
            render=args.render,
            deterministic=deterministic,
        )
        m = episode_metrics(ep)
        all_eps.append(ep)
        all_metrics.append(m)
        print_episode_line(ep_i, args.n_episodes, m)

        if args.save_graphs:
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_timeseries(ep, ep_i, rcfg, bounds, save_dir)

    agg = aggregate_episodes(all_metrics)
    print_summary(agg, bounds, args.algo, args.road)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(agg, all_metrics, args.algo, args.road, args.model_path, save_dir)

    if args.save_graphs:
        print("\n  Generating summary figures ...")
        plot_episode_comparison(all_metrics, rcfg, bounds, save_dir)
        plot_reward_breakdown(all_metrics, save_dir)
        print(f"  All figures saved to {save_dir}")
    else:
        print("  (pass --save-graphs to generate figures)")


if __name__ == "__main__":
    main()

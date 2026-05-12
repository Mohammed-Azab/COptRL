"""
Smoke-tests and conformance checks for QuarterCarEnv.

Run:  python tests/test_env.py
  --render {human,rgb_array,none}  optional: run render rollout instead of tests
"""
import sys
import io
import time
import argparse
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import QuarterCar_env  # registers the environment

LOG_PATH = Path(__file__).parent / "test_env.log"

#  helpers 

def section(title: str, log):
    line = f"\n{'=' * 60}\n{title}\n{'=' * 60}"
    print(line)
    log.write(line + "\n")


def ok(msg: str, log):
    s = f"  [PASS] {msg}"
    print(s); log.write(s + "\n")


def fail(msg: str, log):
    s = f"  [FAIL] {msg}"
    print(s); log.write(s + "\n")


def info(msg: str, log):
    s = f"  {msg}"
    print(s); log.write(s + "\n")


#  individual checks 

def check_registration(log):
    section("1. Environment registration", log)
    env = gym.make("QuarterCar_env/QuarterCar")
    ok("gym.make('QuarterCar_env/QuarterCar') succeeded", log)
    info(f"observation_space : {env.observation_space}", log)
    info(f"action_space      : {env.action_space}", log)
    env.close()


def check_spaces(log):
    section("2. Spaces", log)
    env = gym.make("QuarterCar_env/QuarterCar")
    obs, _ = env.reset()

    assert env.observation_space.shape == (8,), "obs shape"
    ok("observation_space.shape == (8,)", log)

    assert env.action_space.shape == (1,), "action shape"
    ok("action_space.shape == (1,)", log)

    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0
    ok("action_space in [-1, 1]", log)

    assert env.observation_space.contains(obs), "reset obs inside obs space"
    ok("reset() obs is inside observation_space", log)

    env.close()


def check_reset_step(log):
    section("3. reset() / step() contract", log)
    env = gym.make("QuarterCar_env/QuarterCar")
    obs, info_dict = env.reset(seed=0)

    assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
    ok(f"obs.dtype == float32", log)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert isinstance(reward, float), f"reward type {type(reward)}"
    ok("reward is float", log)

    assert isinstance(terminated, bool), f"terminated type {type(terminated)}"
    ok("terminated is bool (not numpy bool)", log)

    assert isinstance(truncated, bool), f"truncated type {type(truncated)}"
    ok("truncated is bool (not numpy bool)", log)

    assert env.observation_space.contains(obs2), "step obs inside obs space"
    ok("step() obs is inside observation_space", log)

    expected_keys = {'rms_accel', 'peak_accel', 'suspension_rms',
                     'comfort_score', 'road_profile', 'step_count', 'episode_time'}
    assert expected_keys.issubset(info2.keys()), f"missing keys: {expected_keys - info2.keys()}"
    ok(f"info dict contains all expected keys", log)

    env.close()


def check_full_episode(log):
    section("4. Full episode rollout (random policy)", log)
    results = {}
    for profile in ["speed_bump", "iso_8608_class_c", "sine_sweep", "flat"]:
        env = gym.make("QuarterCar_env/QuarterCar", road_profile=profile)
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, info_dict = env.step(
                env.action_space.sample()
            )
            total_reward += reward
            steps += 1
            done = terminated or truncated
        env.close()
        results[profile] = {"steps": steps, "return": round(total_reward, 3),
                            "rms_accel": round(info_dict["rms_accel"], 4),
                            "comfort": round(info_dict["comfort_score"], 4)}
        ok(f"{profile:<22}  steps={steps:>4}  return={total_reward:>10.3f}  "
           f"rms_accel={info_dict['rms_accel']:.4f}  comfort={info_dict['comfort_score']:.4f}", log)

    return results


def check_zero_action(log):
    section("5. Passive baseline (zero action = spring-damper only)", log)
    env = gym.make("QuarterCar_env/QuarterCar", road_profile="speed_bump")
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, _, terminated, truncated, info_dict = env.step(np.array([0.0]))
        done = terminated or truncated
    env.close()
    ok(f"Passive episode completed  rms_accel={info_dict['rms_accel']:.4f}  "
       f"comfort={info_dict['comfort_score']:.4f}", log)


def run_gymnasium_check_env(log):
    section("6. gymnasium check_env", log)
    from QuarterCar_env.envs import QuarterCarEnv
    env = QuarterCarEnv()
    buf = io.StringIO()
    warnings_found = []
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            check_env(env, warn=True, skip_render_check=True)
        ok("gymnasium check_env passed with no errors", log)
    except Exception as e:
        fail(f"gymnasium check_env raised: {e}", log)
    captured = buf.getvalue().strip()
    if captured:
        for line in captured.splitlines():
            if "warn" in line.lower() or "UserWarning" in line.lower():
                warnings_found.append(line)
                info(f"  warning: {line}", log)
    if not warnings_found:
        ok("No warnings from gymnasium check_env", log)
    env.close()


def run_sb3_check_env(log):
    section("7. stable-baselines3 check_env", log)
    from QuarterCar_env.envs import QuarterCarEnv
    env = QuarterCarEnv()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            sb3_check_env(env, warn=True, skip_render_check=True)
        ok("SB3 check_env passed with no errors", log)
    except Exception as e:
        fail(f"SB3 check_env raised: {e}", log)
    captured = buf.getvalue().strip()
    if captured:
        for line in captured.splitlines():
            info(f"  {line}", log)
    env.close()


def check_seed_reproducibility(log):
    section("8. Seed reproducibility", log)
    def rollout(seed):
        env = gym.make("QuarterCar_env/QuarterCar", road_profile="speed_bump")
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(20):
            obs, r, terminated, truncated, _ = env.step(np.array([0.0]))
            rewards.append(r)
            if terminated or truncated:
                break
        env.close()
        return rewards

    r1 = rollout(7)
    r2 = rollout(7)
    r3 = rollout(99)

    if r1 == r2:
        ok("Same seed → identical rewards", log)
    else:
        fail("Same seed produced different rewards", log)

    if r1 != r3:
        ok("Different seeds → different rewards", log)
    else:
        fail("Different seeds produced identical rewards (suspicious)", log)


#  render rollout ─

def run_render_rollout(render_mode: str):
    """
    5-second rollout with sinusoidal F_D policy.
    Road: speed_bump, bump 10 m ahead of start, A=0.08 m, L=1.0 m.
    """
    from QuarterCar_env.envs import QuarterCarEnv
    from QuarterCar_env.params import F_MAX, DT, VEHICLE_SPEED

    speed    = VEHICLE_SPEED   # 10 m/s
    n_steps  = int(5.0 / DT)   # 250 steps = 5 s
    period   = 1.0             # sinusoidal period (s)
    amp      = 0.4 * F_MAX     # sinusoidal amplitude

    env = QuarterCarEnv(
        road_profile='speed_bump',
        vehicle_speed=speed,
        render_mode=render_mode,
        road_params={
            'bump_x_start': 10.0,   # bump 10 m ahead
            'bump_length':   3.5,   # 1 m wide for visibility
            'bump_height':   0.15,  # 8 cm
        },
    )
    obs, _ = env.reset(seed=0)

    frames = []
    t_wall_start = time.perf_counter()

    for step_i in range(n_steps):
        t_sim  = step_i * DT
        action = np.array([amp * np.sin(2.0 * np.pi * t_sim / period) / F_MAX],
                          dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)

        if render_mode == 'rgb_array':
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t_wall_start
    fps = (step_i + 1) / elapsed
    print(f"\nRender FPS (wall-time): {fps:.1f}  ({step_i+1} frames in {elapsed:.2f} s)")

    if render_mode == 'rgb_array' and frames:
        out_dir = Path(__file__).parent.parent / 'outputs'
        out_dir.mkdir(exist_ok=True)
        try:
            import imageio
            try:
                import imageio_ffmpeg   # noqa: F401
                out_path = out_dir / 'rollout.mp4'
                imageio.mimsave(str(out_path), frames,
                                fps=env.metadata['render_fps'])
            except ImportError:
                out_path = out_dir / 'rollout.gif'
                imageio.mimsave(str(out_path), frames,
                                fps=env.metadata['render_fps'])
            print(f"Saved: {out_path.resolve()}  ({out_path.stat().st_size / 1024:.1f} KB)")
        except ImportError:
            print("imageio not available - skipping file save.")

    env.close()


#  main ─

def main():
    with open(LOG_PATH, "w") as log:
        header = f"QuarterCarEnv test run\nPython {sys.version}\n"
        print(header); log.write(header + "\n")

        checks = [
            check_registration,
            check_spaces,
            check_reset_step,
            check_full_episode,
            check_zero_action,
            run_gymnasium_check_env,
            run_sb3_check_env,
            check_seed_reproducibility,
        ]

        passed = failed = 0
        for fn in checks:
            try:
                fn(log)
                passed += 1
            except Exception as e:
                msg = f"\n  [ERROR] {fn.__name__} raised:\n  {traceback.format_exc()}"
                print(msg); log.write(msg + "\n")
                failed += 1

        summary = f"\n{'=' * 60}\nSummary: {passed} passed, {failed} failed\n{'=' * 60}\n"
        print(summary); log.write(summary)

    print(f"\nFull output saved to {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuarterCarEnv test / render demo")
    parser.add_argument(
        '--render',
        nargs='?',
        const='human',
        default=None,
        choices=['human', 'rgb_array', 'none'],
        help="Run render rollout (default: run existing conformance tests)",
    )
    args = parser.parse_args()

    if args.render is None or args.render == 'none':
        main()
    else:
        run_render_rollout(args.render)

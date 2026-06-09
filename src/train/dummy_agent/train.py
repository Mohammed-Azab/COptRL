# constant-speed dummy agent

import argparse
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym

_src = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_src / "gym_env"))
sys.path.insert(0, str(_src))      
import QuarterCar_env.envs  # noqa: F401   

ZERO_ACTION = np.array([0.0], dtype=np.float32)  # no acceleration = constant speed
SPEEDING_UP = np.array([1.0], dtype=np.float32)  
SLOWING_DOWN = np.array([-1.0], dtype=np.float32)  


def run_episodes(road: str, n_episodes: int, render: bool) -> None:
    env = gym.make(
        "QuarterCar_env/QuarterCar",
        road_profile=road,
        render_mode="human" if render else "none",
    )

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            obs, reward, terminated, truncated, info = env.step(SPEEDING_UP)
            done = terminated or truncated
            ep_reward += reward
            step += 1

        print(
            f"Ep {ep:3d} | steps={step:4d} | reward={ep_reward:8.2f} "
            f"| rms_a={info.get('rms_accel', 0.0):.3f} m/s² "
            f"| speed={info.get('speed', 0.0):.2f} m/s"
        )
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Constant-speed dummy agent")
    parser.add_argument(
                        "--road",
                        default="speed_bump",
                        choices=["speed_bump", "flat", "recorded"])
    parser.add_argument(
                        "--episodes",
                        type=int,
                        default=5)
    parser.add_argument(
                        "--render",
                        action="store_true")
    args = parser.parse_args()

    run_episodes(args.road, args.episodes, args.render)


if __name__ == "__main__":
    main()

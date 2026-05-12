"""Dummy-agent runner and SB3 trainer for the quarter-car active suspension environment """

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from QuarterCar_env.envs import QuarterCarEnv
from QuarterCar_env.params import V_MAX, V_MIN

#   tuning parameters
H_THRESH = 0.01   # m - road height threshold 
ACCEL    = 0.01   # normalised speed increment per step when no bump detected
K_SUSP   = 8.0    # PD proportional gain: travel → normalised suspension force
B_SUSP   = 0.8    # PD derivative gain: relative body–wheel velocity → F_norm


def _action(
    obs: np.ndarray,
    mode: str,
    v_cmd: float,
    v_max: float,
) -> tuple[np.ndarray, float]:
    """Return (action, updated_v_cmd) from the hand-coded  policy."""
    road_height = float(obs[4])
    travel      = float(obs[6])   # suspension travel = z_W − z_B
    body_vel    = float(obs[1])   # ż_B
    wheel_vel   = float(obs[3])   # ż_W

    F_norm = -K_SUSP * travel - B_SUSP * (body_vel - wheel_vel)
    F_norm = float(np.clip(F_norm, -1.0, 1.0))

    if road_height > H_THRESH:
        v_cmd = V_MIN / v_max
    else:
        v_cmd = min(1.0, v_cmd + ACCEL)

    if mode == "suspension":
        return np.array([F_norm], dtype=np.float32), v_cmd
    if mode == "speed":
        return np.array([v_cmd], dtype=np.float32), v_cmd
    # "hybrid"
    return np.array([F_norm, v_cmd], dtype=np.float32), v_cmd


def run_episodes(
    mode: str,
    road: str,
    n_episodes: int,
    render: bool,
    v_max: float = V_MAX,
) -> tuple[list[dict], dict]:
    """Run N  episodes. Returns (results list, last-episode time-series)."""
    env = QuarterCarEnv(
        road_profile=road,
        control_mode=mode,
        v_max=v_max,
        render_mode='human' if render else 'none',
    )

    results: list[dict] = []
    last_ep_data: dict = {}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done      = False
        ep_return = 0.0
        speeds: list[float]     = []
        speed_errs: list[float] = []
        v_cmd = 1.0

        ep_data: dict[str, list] = {
            't': [], 'accel': [], 'speed': [], 'road_h': [], 'reward': [],
        }

        while not done:
            action, v_cmd = _action(obs, mode, v_cmd, v_max)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward

            ep_data['t'].append(info.get('episode_time', 0.0))
            ep_data['accel'].append(info.get('rms_accel', 0.0))
            ep_data['reward'].append(reward)
            ep_data['road_h'].append(float(obs[4]))

            if mode in ('speed', 'hybrid'):
                spd = info.get('speed', float(obs[8]) if len(obs) > 8 else 0.0)
                ep_data['speed'].append(spd)
                speeds.append(spd)
                speed_errs.append(abs(info.get('speed_error', 0.0)))
            else:
                ep_data['speed'].append(0.0)

        rms_a   = float(info.get('rms_accel',  0.0))
        peak_a  = float(info.get('peak_accel', 0.0))
        m_speed = float(np.mean(speeds)) if speeds else 0.0
        sp_rms  = float(np.sqrt(np.mean(np.square(speed_errs)))) if speed_errs else 0.0

        results.append({
            'episode':         ep,
            'return':          round(ep_return, 4),
            'rms_accel':       round(rms_a, 4),
            'peak_accel':      round(peak_a, 4),
            'mean_speed':      round(m_speed, 4),
            'speed_error_rms': round(sp_rms, 4),
            'road_profile':    road,
        })
        print(
            f"Ep {ep:3d} | return={ep_return:8.2f} | rms_a={rms_a:.3f} m/s² "
            f"| peak_a={peak_a:.3f} m/s² | mean_v={m_speed:.2f} m/s "
            f"| sp_err_rms={sp_rms:.3f} m/s"
        )
        last_ep_data = ep_data

    env.close()
    return results, last_ep_data

def main():
    parser = argparse.ArgumentParser(
        description=' runner + optional SB3 trainer for quarter-car env')
    parser.add_argument('--mode',      default='suspension',
                        choices=['suspension', 'speed', 'hybrid'])
    parser.add_argument('--road',      default='speed_bump',
                        choices=['speed_bump', 'iso_8608_class_c', 'sine_sweep', 'flat'])
    parser.add_argument('--episodes',  type=int, default=10,
                        help=' evaluation episodes.')
    parser.add_argument('--render',    action='store_true')
    args = parser.parse_args()

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # baseline 
    results, _ = run_episodes(args.mode, args.road, args.episodes, args.render)

    returns  = [r['return']          for r in results]
    rms_accs = [r['rms_accel']       for r in results]
    pk_accs  = [r['peak_accel']      for r in results]
    mean_vs  = [r['mean_speed']      for r in results]
    sp_errs  = [r['speed_error_rms'] for r in results]

    print('\n Aggregate ')
    print(f"  Mean return        : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean RMS accel     : {np.mean(rms_accs):.3f} m/s²")
    print(f"  Mean peak accel    : {np.mean(pk_accs):.3f} m/s²")
    print(f"  Mean speed         : {np.mean(mean_vs):.2f} m/s")
    print(f"  Speed error RMS    : {np.mean(sp_errs):.3f} m/s")

    csv_path = results_dir / f'dummy_{args.mode}_{args.road}_{ts_str}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\n results → {csv_path}')
if __name__ == '__main__':
    main()

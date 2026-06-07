"""
QuarterCarEnv — Gymnasium environment for vehicle speed control over road bumps.

Action  (1,) float32  in [-1, 1]
  a_cmd = action[0] * a_max  [m/s²]
  v_{t+1} = clip(v_t + a_cmd * DT, 0, v_max)

Observation  (6 + 3*n_peaks,) float32  — base env outputs 6 scalars;
  PreviewWrapper appends n_peaks peak slots on top.

  [0]  ζ            road height at wheel          [m]
  [1]  ζ̇           road velocity at wheel         [m/s]
  [2]  v / v_max    normalised longitudinal speed  [0, 1]
  [3]  filtered_a / a_comfort                      smoothed long. accel
  [4]  filtered_jerk / j_max                       smoothed jerk
  [5]  prev_action                                 last action sent

  [6 … 6+3n-1]  peak preview from PreviewWrapper
    each peak: [dist/D, height/h_clip, width/D]  in [0, 1]
    missing peaks fill with [1.0, 0.0, 0.0] (bump at horizon)

Wrap with PreviewWrapper before VecEnv to get the full observation.
"""

from gymnasium.envs.registration import register
from QuarterCar_env.envs.quarter_car_env import QuarterCarEnv

register(
    id="QuarterCar_env/QuarterCar",
    entry_point="QuarterCar_env.envs:QuarterCarEnv",
)

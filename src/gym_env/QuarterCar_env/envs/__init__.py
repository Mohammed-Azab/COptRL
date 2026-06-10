# QuarterCarEnv: Gymnasium env for speed control over road bumps.
#
# Action: (1,) float32 in [-1, 1]
#   a_cmd = action[0] * a_max  [m/s²]
#   v_next = clip(v + a_cmd * DT, 0, v_max)
#
# Observation: (6 + 3*n_peaks,) float32
#   [0]  ζ          road height at wheel        [m]
#   [1]  ζ̇         road velocity at wheel       [m/s]
#   [2]  v/v_max    normalised longitudinal speed [0, 1]
#   [3]  filtered_a/a_comfort
#   [4]  filtered_jerk/j_max
#   [5]  prev_action
#   [6…] peak preview slots added by PreviewWrapper (dist, height, width) in [0,1]
#        missing peaks fill with [1.0, 0.0, 0.0]
#
# Wrap with PreviewWrapper before VecEnv to get the full observation.

from gymnasium.envs.registration import register
from QuarterCar_env.envs.quarter_car_env import QuarterCarEnv

register(
    id="QuarterCar_env/QuarterCar",
    entry_point="QuarterCar_env.envs:QuarterCarEnv",
)

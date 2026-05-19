"""
QuarterCarEnv: A Gymnasium environment for quarter-car active suspension RL.

State (internal, 6-D float64):
  x[0] = ζ − z_W   tire deflection           [m]
  x[1] = ż_W       wheel vertical velocity    [m/s]
  x[2] = z_W − z_B suspension travel          [m]
  x[3] = ż_B       body vertical velocity     [m/s]
  x[4] = v         longitudinal velocity      [m/s]
  x[5] = z_B       body displacement from static eq. [m]

Observation (8,) float32 - clipped to OBS_LOW/OBS_HIGH:
  idx 0: z_B            body displacement      [m]
  idx 1: ż_B            body velocity          [m/s]
  idx 2: z_W = z_B + (z_W−z_B)  wheel displacement [m]
  idx 3: ż_W            wheel velocity         [m/s]
  idx 4: ζ              road height            [m]
  idx 5: ζ̇             road velocity          [m/s]
  idx 6: z_W − z_B      suspension travel      [m]
  idx 7: ζ − z_W        tire deflection        [m]

Action (1,) float32 ∈ [−1, 1]:
  F_act = action[0] × F_MAX   (F_MAX = 10 000 N)
  Positive → lifts body / presses wheel down; applied equal-and-opposite.
"""

from QuarterCar_env.envs.quarter_car_env import QuarterCarEnv

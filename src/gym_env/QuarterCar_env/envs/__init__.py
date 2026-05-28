"""
QuarterCarEnv: A Gymnasium environment for vehicle speed control over road bumps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION  (1,) float32  ∈ [−1, 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  a_cmd = action[0] × a_max   [m/s²]
  v_{t+1} = clip(v_t + a_cmd × DT, 0, v_max)

  +1 → maximum acceleration
  −1 → maximum braking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBSERVATION  (16,) float32  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ── Road contact ───────
  [0]   ζ            road height at wheel          [m]    clip ±0.15
  [1]   ζ̇           road velocity at wheel         [m/s]  clip ±7.00

  ── Speed ───────────
  [2]   v / v_max    normalised longitudinal speed      ∈ [0, 1]

  ── Comfort context ───
  [3]   filtered_a / a_comfort   smoothed accel         if obs_enable_accel
  [4]   filtered_jerk / j_max    smoothed jerk          if obs_enable_jerk
  [5]   prev_action              last action sent       if obs_enable_prev_action

  ── Road preview — what is coming ahead 
  [6]   preview[0]   road height  2m ahead (normalised to ±1)
  [7]   preview[1]   road height  4m ahead
  [8]   preview[2]   road height  6m ahead
  [9]   preview[3]   road height  8m ahead
  [10]  preview[4]   road height 10m ahead
  [11]  preview[5]   road height 12m ahead
  [12]  preview[6]   road height 14m ahead
  [13]  preview[7]   road height 16m ahead
  [14]  preview[8]   road height 18m ahead
  [15]  preview[9]   road height 20m ahead

  Spacing = preview_distance / n_preview_points  (default: 20m / 10 = 2m).
  Each value is clipped to ±preview_height_clip then divided by it → ±1.
  Preview is always at the END of obs; indices [6:] shift if n_preview_points
  or the comfort-context toggles change.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USING THE PREVIEW IN THE POLICY NETWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use PreviewFeaturesExtractor (src/train/preview_extractor.py) to give the
agent a spatial prior over the bump profile.  It splits the obs at the
preview boundary, runs obs[0:6] through an MLP and obs[6:16] through a
1D Conv, then merges both branches.

  from train.preview_extractor import PreviewFeaturesExtractor

  policy_kwargs = dict(
      features_extractor_class=PreviewFeaturesExtractor,
      features_extractor_kwargs=dict(
          n_preview_points=10,   # must match reward_params.yaml
          state_hidden_dim=64,
          conv_channels=8,
          features_dim=128,
      ),
      net_arch=[128, 128],
  )
  model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)

  WARNING: n_preview_points must equal reward_params.yaml n_preview_points.
  A mismatch silently feeds wrong values into each branch.
"""

from gymnasium.envs.registration import register
from QuarterCar_env.envs.quarter_car_env import QuarterCarEnv

register(
    id="QuarterCar_env/QuarterCar",
    entry_point="QuarterCar_env.envs:QuarterCarEnv",
)

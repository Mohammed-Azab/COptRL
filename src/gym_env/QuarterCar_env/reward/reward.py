"""
Comfort-aware reward system for the QuarterCar speed-planning environment.

Reward composition:
    R = w_tracking * r_tracking
      + w_accel    * r_accel
      + w_jerk     * r_jerk
      + w_action_smooth * r_action_smooth
      + w_curve    * r_curve
      + w_energy   * r_energy

All terms are normalized to roughly [-1, 0] before weighting.
References: [15][17][18][19] in refs.txt
"""

import numpy as np
from typing import Tuple

from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config


# ---------------------------------------------------------------------------
# Pure term functions — each returns a float in roughly [-inf, 0]
# ---------------------------------------------------------------------------

def r_tracking(v: float, v_target: float, v_max: float) -> float:
    """Velocity tracking penalty. Returns 0 when v == v_target, -1 when |error| == v_max."""
    return -((v - v_target) / v_max) ** 2


def r_accel(a: float, a_comfort: float, accel_clip: float) -> float:
    """Longitudinal comfort penalty. Returns -1 when |a| == a_comfort. ISO 2631 aligned."""
    a_c = float(np.clip(a, -accel_clip, accel_clip))
    return -(a_c / a_comfort) ** 2


def r_jerk(jerk: float, j_max: float, jerk_clip: float) -> float:
    """Jerk penalty. Returns -1 when |jerk| == j_max."""
    j_c = float(np.clip(jerk, -jerk_clip, jerk_clip))
    return -(j_c / j_max) ** 2


def r_action_smooth(u_t: float, u_prev: float) -> float:
    """Action smoothness penalty. Penalises sudden command changes. Returns 0 when unchanged."""
    return -(u_t - u_prev) ** 2


def r_curve(v: float, curvature: float, a_lat_max: float, curvature_clip: float) -> float:
    """Lateral comfort penalty from road curvature. a_lat = v^2 x |curvature|."""
    k = float(np.clip(curvature, -curvature_clip, curvature_clip))
    a_lat = (v ** 2) * abs(k)
    return -(a_lat / a_lat_max) ** 2


def r_energy(u: float) -> float:
    """Control effort penalty on normalised action. Returns -1 at full deflection."""
    return -(u ** 2)

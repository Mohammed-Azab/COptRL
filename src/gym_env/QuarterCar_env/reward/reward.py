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


# ---------------------------------------------------------------------------
# Speed-target helper
# ---------------------------------------------------------------------------

def compute_v_target(v_ref: float, mode: str, curvature: float, cfg: RewardConfig) -> float:
    """
    Compute the reward tracking target speed.

    In "curvature_aware" mode the curvature-safe speed limit is min'd with v_ref.
    In "constant" and "external" mode v_ref is returned unchanged.
    """
    if mode == "curvature_aware":
        curve_limit = float(np.sqrt(cfg.a_lat_comfort / (abs(curvature) + 1e-6)))
        curve_limit = float(np.clip(curve_limit, cfg.min_curve_speed, cfg.max_curve_speed))
        return min(v_ref, curve_limit)
    return v_ref


# ---------------------------------------------------------------------------
# Reward assembler
# ---------------------------------------------------------------------------

def compute_reward(
    v: float,
    v_target: float,
    a_actual: float,
    filtered_a: float,
    jerk: float,
    filtered_jerk: float,
    prev_action: float,
    action: float,
    curvature: float,
    cfg: RewardConfig,
) -> tuple[float, dict]:
    """
    Assemble total reward from individual terms.

    Returns (total_reward, breakdown) where breakdown is a flat dict with every
    term value plus "reward_total". Insert directly into the env's info dict.

    Args:
        v:            Current speed [m/s].
        v_target:     Target speed for tracking term [m/s].
        a_actual:     Raw finite-difference acceleration [m/s²].
        filtered_a:   IIR-smoothed acceleration [m/s²] — used for r_accel.
        jerk:         Raw finite-difference jerk [m/s³].
        filtered_jerk: IIR-smoothed jerk [m/s³] — used for r_jerk.
        prev_action:  Previous normalised action in [-1, 1].
        action:       Current normalised action in [-1, 1].
        curvature:    Road curvature [m^-1].
        cfg:          RewardConfig.
    """
    bd: dict = {}
    total = 0.0

    if cfg.enable_tracking:
        rt = r_tracking(v, v_target, cfg.v_max)
        bd["r_tracking"] = rt
        total += cfg.w_tracking * rt
    else:
        bd["r_tracking"] = 0.0

    if cfg.enable_accel:
        ra = r_accel(filtered_a, cfg.a_comfort, cfg.accel_clip)
        bd["r_accel"] = ra
        total += cfg.w_accel * ra
    else:
        bd["r_accel"] = 0.0

    if cfg.enable_jerk:
        rj = r_jerk(filtered_jerk, cfg.j_max, cfg.jerk_clip)
        bd["r_jerk"] = rj
        total += cfg.w_jerk * rj
    else:
        bd["r_jerk"] = 0.0

    if cfg.enable_action_smooth:
        rs = r_action_smooth(action, prev_action)
        bd["r_action_smooth"] = rs
        total += cfg.w_action_smooth * rs
    else:
        bd["r_action_smooth"] = 0.0

    if cfg.enable_curve:
        rc = r_curve(v, curvature, cfg.a_lat_max, cfg.curvature_clip)
        bd["r_curve"] = rc
        total += cfg.w_curve * rc
    else:
        bd["r_curve"] = 0.0

    if cfg.enable_energy:
        re = r_energy(action)
        bd["r_energy"] = re
        total += cfg.w_energy * re
    else:
        bd["r_energy"] = 0.0

    # Guard against NaN/Inf from extreme inputs
    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))
    for key in bd:
        bd[key] = float(np.nan_to_num(bd[key], nan=0.0, posinf=0.0, neginf=0.0))

    bd["reward_total"] = total
    return total, bd


# ---------------------------------------------------------------------------
# Terminal bonus
# ---------------------------------------------------------------------------

def compute_terminal_bonus(rms_accel: float, cfg: RewardConfig) -> float:
    """Return terminal_bonus if rms_accel < a_limit, else terminal_penalty."""
    if rms_accel < cfg.a_limit:
        return cfg.terminal_bonus
    return cfg.terminal_penalty

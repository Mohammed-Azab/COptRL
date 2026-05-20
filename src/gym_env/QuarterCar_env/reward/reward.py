"""
Comfort-aware reward system for the QuarterCar speed-planning environment.

Reward composition:
    R = w_comfort_bonus * r_comfort_bonus  — positive per-step reward for smooth riding
      + w_tracking      * r_tracking       — stay near the reference speed
      + w_accel         * r_accel          — penalise harsh longitudinal acceleration (ISO 2631)
      + w_jerk          * r_jerk           — penalise rapid acceleration changes
      + w_action_smooth * r_action_smooth  — penalise discontinuous commands
      + w_curve         * r_curve          — penalise lateral discomfort from road curvature

Step reward range with default config:
    per-step  ∈ [-5.0,  +0.8]
    episode   ∈ [-1350, +300]   (theoretical; practical bad-agent range ≈ -500 to 0)

Episode max (+300) is achieved by: perfect tracking + zero accel + terminal bonus.
References: [15][17][18][19] in refs.txt
"""

import numpy as np

from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config


def r_tracking(v: float, v_target: float, v_max: float) -> float:
    """Velocity tracking penalty. Returns 0 when v == v_target, -1 when |error| == v_max."""
    return -((v - v_target) / v_max) ** 2


def r_accel(a: float, a_comfort: float, accel_clip: float) -> float:
    """Longitudinal comfort penalty. Returns -1 when |a| == a_comfort. (ISO 2631)."""
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


def r_comfort_bonus(filtered_a: float, a_comfort: float) -> float:
    """Positive per-step reward for riding inside the comfort envelope.

    Returns +1 when filtered_a == 0 (perfectly smooth ride),
    0 when |filtered_a| == a_comfort (at the comfort boundary),
    and is clipped to 0 beyond — r_accel handles the penalty side.
    """
    return max(0.0, 1.0 - (filtered_a / a_comfort) ** 2)


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


def reward_bounds(cfg: RewardConfig, n_steps: int) -> dict:
    """
    Theoretical per-step and episode reward bounds for a given RewardConfig.

    Per-step max: w_comfort_bonus × 1.0 when accel == 0 and all penalties are zero.
    Per-step min: all weighted penalty terms at their reward clip boundaries.
      Uses reward_accel_clip / reward_jerk_clip (not the obs clips) for the bound.

    Episode bounds stack the terminal bonus/penalty on top of n_steps × per-step.
    The episode_min is a hard mathematical limit — in practice the IIR filters
    prevent simultaneous worst-case on every step, so real bad agents stay
    well above episode_min (typical random-agent range is roughly -500 to 0).
    """
    per_step_max = 0.0
    per_step_min = 0.0

    if cfg.enable_comfort_bonus:
        # best: filtered_a == 0  → r_comfort_bonus = 1
        per_step_max += cfg.w_comfort_bonus * 1.0

    if cfg.enable_tracking:
        # worst: |v - v_target| == v_max  → r_tracking = -1
        per_step_min += cfg.w_tracking * (-1.0)

    if cfg.enable_accel:
        # worst: |filtered_a| == reward_accel_clip  → r_accel = -(clip/a_comfort)²
        per_step_min += cfg.w_accel * -(cfg.reward_accel_clip / cfg.a_comfort) ** 2

    if cfg.enable_jerk:
        # worst: |filtered_jerk| == reward_jerk_clip  → r_jerk = -(clip/j_max)²
        per_step_min += cfg.w_jerk * -(cfg.reward_jerk_clip / cfg.j_max) ** 2

    if cfg.enable_action_smooth:
        # worst: |u_t - u_prev| == 2  (full swing from -1 to +1) → r_smooth = -4
        per_step_min += cfg.w_action_smooth * (-4.0)

    if cfg.enable_curve:
        # worst: v == v_max, |curvature| == curvature_clip → a_lat = v_max² × curvature_clip
        worst_a_lat = (cfg.v_max ** 2) * cfg.curvature_clip
        per_step_min += cfg.w_curve * -(worst_a_lat / cfg.a_lat_max) ** 2

    return {
        "per_step_max": round(per_step_max, 6),
        "per_step_min": round(per_step_min, 6),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus,   4),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 4),
        "n_steps": n_steps,
    }


def compute_terminal_bonus(rms_accel: float, cfg: RewardConfig) -> float:
    """Grant a bonus or penalty at episode end based on ride comfort (RMS body accel)."""
    if rms_accel < cfg.a_limit:
        return cfg.terminal_bonus
    return cfg.terminal_penalty


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
        v:             Current speed [m/s].
        v_target:      Target speed for tracking term [m/s].
        a_actual:      Raw finite-difference acceleration [m/s²].
        filtered_a:    IIR-smoothed acceleration [m/s²] — used for r_accel.
        jerk:          Raw finite-difference jerk [m/s³].
        filtered_jerk: IIR-smoothed jerk [m/s³] — used for r_jerk.
        prev_action:   Previous normalised action in [-1, 1].
        action:        Current normalised action in [-1, 1].
        curvature:     Road curvature [m^-1].
        cfg:           RewardConfig.
    """
    bd: dict = {}
    total = 0.0

    if cfg.enable_tracking:
        rt = r_tracking(v, v_target, cfg.v_max)
        bd["r_tracking"] = rt
        total += cfg.w_tracking * rt
    else:
        bd["r_tracking"] = 0.0

    if cfg.enable_comfort_bonus:
        rb = r_comfort_bonus(filtered_a, cfg.a_comfort)
        bd["r_comfort_bonus"] = rb
        total += cfg.w_comfort_bonus * rb
    else:
        bd["r_comfort_bonus"] = 0.0

    if cfg.enable_accel:
        # reward_accel_clip is tighter than the obs clip — limits worst-case penalty
        ra = r_accel(filtered_a, cfg.a_comfort, cfg.reward_accel_clip)
        bd["r_accel"] = ra
        total += cfg.w_accel * ra
    else:
        bd["r_accel"] = 0.0

    if cfg.enable_jerk:
        # reward_jerk_clip is tighter than the obs clip — limits worst-case penalty
        rj = r_jerk(filtered_jerk, cfg.j_max, cfg.reward_jerk_clip)
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

    # guard against NaN/Inf from extreme inputs
    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))
    for key in bd:
        bd[key] = float(np.nan_to_num(bd[key], nan=0.0, posinf=0.0, neginf=0.0))

    bd["reward_total"] = total
    return total, bd

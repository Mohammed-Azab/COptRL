"""
Reward composition:
    R = w_comfort_bonus * r_comfort_bonus
      + w_tracking      * r_speed_band
      + w_accel         * r_accel
      + w_jerk          * r_jerk
      + w_action_smooth * r_action_smooth
"""

import numpy as np

from QuarterCar_env.config.reward_params import RewardConfig

def r_speed_band(v: float, v_min: float, v_upper: float) -> float:

    if v < v_min:
        return -((v_min - v) / v_min) ** 2
    if v > v_upper:
        return -((v - v_upper) / v_upper) ** 2
    return 0.0

def r_accel(a: float, a_comfort: float, accel_clip: float) -> float:
    # Longitudinal accelration 
    a_c = float(np.clip(a, -accel_clip, accel_clip))
    return -(a_c / a_comfort) ** 2

def r_jerk(jerk: float, j_max: float, jerk_clip: float) -> float:
    # Jerk penalty.
    j_c = float(np.clip(jerk, -jerk_clip, jerk_clip))
    return -(j_c / j_max) ** 2

def r_action_smooth(u_t: float, u_prev: float) -> float:
    # Action smoothness 
    return -(u_t - u_prev) ** 2

def r_comfort_bonus(filtered_a: float, a_comfort: float) -> float:
    # Positive per-step reward
    return max(0.0, 1.0 - (filtered_a / a_comfort) ** 2)

def compute_terminal_bonus(rms_accel: float, cfg: RewardConfig) -> float:
    # Terminal Reward based on ride comfort (RMS body accel)
    if rms_accel < cfg.a_limit:
        return cfg.terminal_bonus
    return cfg.terminal_penalty

def compute_reward(
    v: float,
    v_upper: float,
    filtered_a: float,
    filtered_jerk: float,
    prev_action: float,
    action: float,
    cfg: RewardConfig,
) -> tuple[float, dict]:

    # Returns (total_reward, breakdown)
    bd: dict = {}
    total = 0.0

    if cfg.enable_tracking:
        rt = r_speed_band(v, cfg.v_min, v_upper)
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
        ra = r_accel(filtered_a, cfg.a_comfort, cfg.reward_accel_clip)
        bd["r_accel"] = ra
        total += cfg.w_accel * ra
    else:
        bd["r_accel"] = 0.0

    if cfg.enable_jerk:
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

    bd["r_curve"] = 0.0

    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))
    for key in bd:
        bd[key] = float(np.nan_to_num(bd[key], nan=0.0, posinf=0.0, neginf=0.0))

    bd["reward_total"] = total
    return total, bd

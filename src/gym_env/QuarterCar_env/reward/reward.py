"""
Reward composition:
    R_step = (v / v_max) * (
        w_heave * r_heave + w_wheel * r_wheel
        + w_tracking * r_tracking + w_accel * r_accel
        + w_jerk * r_jerk + w_action_smooth * r_action_smooth
    )
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
    a_c = float(np.clip(a, -accel_clip, accel_clip))
    return -(a_c / a_comfort) ** 2


def r_jerk(jerk: float, j_max: float, jerk_clip: float) -> float:
    j_c = float(np.clip(jerk, -jerk_clip, jerk_clip))
    return -(j_c / j_max) ** 2


def r_action_smooth(u_t: float, u_prev: float) -> float:
    return -(u_t - u_prev) ** 2


def r_heave(z_B_ddot: float, a_B_comfort: float) -> float:
    return -(z_B_ddot / a_B_comfort) ** 2


def r_wheel(z_W_ddot: float, a_W_comfort: float) -> float:
    return -(z_W_ddot / a_W_comfort) ** 2


def compute_terminal_bonus(rms_accel: float, cfg: RewardConfig) -> float:
    if rms_accel < cfg.a_limit:
        return cfg.terminal_bonus
    return cfg.terminal_penalty


def compute_reward(
    v: float,
    v_upper: float,
    z_B_ddot: float,
    z_W_ddot: float,
    filtered_a: float,
    filtered_jerk: float,
    prev_action: float,
    action: float,
    cfg: RewardConfig,
) -> tuple[float, dict]:

    bd: dict = {}
    core = 0.0

    if cfg.enable_heave:
        rh = r_heave(z_B_ddot, cfg.a_B_comfort)
        bd["r_heave"] = rh
        core += cfg.w_heave * rh
    else:
        bd["r_heave"] = 0.0

    if cfg.enable_wheel:
        rw = r_wheel(z_W_ddot, cfg.a_W_comfort)
        bd["r_wheel"] = rw
        core += cfg.w_wheel * rw
    else:
        bd["r_wheel"] = 0.0

    if cfg.enable_tracking:
        rt = r_speed_band(v, cfg.v_min, v_upper)
        bd["r_tracking"] = rt
        core += cfg.w_tracking * rt
    else:
        bd["r_tracking"] = 0.0

    if cfg.enable_accel:
        ra = r_accel(filtered_a, cfg.a_comfort, cfg.reward_accel_clip)
        bd["r_accel"] = ra
        core += cfg.w_accel * ra
    else:
        bd["r_accel"] = 0.0

    if cfg.enable_jerk:
        rj = r_jerk(filtered_jerk, cfg.j_max, cfg.reward_jerk_clip)
        bd["r_jerk"] = rj
        core += cfg.w_jerk * rj
    else:
        bd["r_jerk"] = 0.0

    if cfg.enable_action_smooth:
        rs = r_action_smooth(action, prev_action)
        bd["r_action_smooth"] = rs
        core += cfg.w_action_smooth * rs
    else:
        bd["r_action_smooth"] = 0.0

    scale = float(np.clip(v / cfg.v_max, 0.0, 1.0)) if cfg.enable_vel_scaling else 1.0
    total = scale * core

    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))
    for key in bd:
        bd[key] = float(np.nan_to_num(bd[key], nan=0.0, posinf=0.0, neginf=0.0))

    bd["r_curve"]      = 0.0
    bd["reward_total"] = total
    return total, bd

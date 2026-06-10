import numpy as np

from QuarterCar_env.config.reward_params import RewardConfig


def j_heave(z_B_ddot: float, g: float, heave_clip: float) -> float:
    z_c = float(np.clip(z_B_ddot, -heave_clip, heave_clip))
    return -(z_c / g) ** 2


def j_wheel(z_W_ddot: float, g: float, wheel_clip: float) -> float:
    z_c = float(np.clip(z_W_ddot, -wheel_clip, wheel_clip))
    return -(z_c / g) ** 2


def j_long(a: float, g: float, accel_clip: float) -> float:
    a_c = float(np.clip(a, -accel_clip, accel_clip))
    return -(a_c / g) ** 2


def j_speed(v: float, v_min: float, v_ref: float) -> float:
    if v < v_min:
        return -1.0 - ((v_min - v) / v_min) ** 2
    return -abs(v_ref - v) / max(v_ref, 0.1)


def j_jerk(jerk: float, j_max: float, jerk_clip: float) -> float:
    j_c = float(np.clip(jerk, -jerk_clip, jerk_clip))
    return -(j_c / j_max) ** 2


def j_action_smooth(u_t: float, u_prev: float) -> float:
    return -(u_t - u_prev) ** 2


def r_progress(s_pos: float, road_length: float) -> float:
    if road_length <= 0.0:
        return 0.0
    return float(np.clip(s_pos / road_length, 0.0, 1.0))


def compute_terminal_bonus(rms_accel: float, mean_speed: float, cfg: RewardConfig) -> float:
    if rms_accel < cfg.a_limit and mean_speed >= cfg.v_min:
        return cfg.terminal_bonus
    return cfg.terminal_penalty


def compute_reward(
    v: float,
    z_B_ddot: float,
    z_W_ddot: float,
    filtered_a: float,
    filtered_jerk: float,
    prev_action: float,
    action: float,
    cfg: RewardConfig,
    s_pos: float = 0.0,
    road_length: float = 0.0,
    bump_ends: list = [],
    bumps_passed: int = 0,
    v_ref: float = 0.0,
) -> tuple[float, dict, int]:

    _v_ref = v_ref if v_ref > 0.0 else cfg.v_max

    Jh = j_heave(z_B_ddot, cfg.g, cfg.reward_heave_clip)
    Jw = j_wheel(z_W_ddot, cfg.g, cfg.reward_wheel_clip)
    Jl = j_long(filtered_a, cfg.g, cfg.reward_accel_clip)
    core = cfg.Q_zBddot * Jh + cfg.Q_zWddot * Jw + cfg.Q_a * Jl

    Js = j_speed(v, cfg.v_min, _v_ref)
    tracking_penalty = cfg.Q_v * Js

    Jj = j_jerk(filtered_jerk, cfg.j_max, cfg.reward_jerk_clip)
    Jsm = j_action_smooth(action, prev_action)
    jerk_smooth_penalty = cfg.w_jerk * Jj + cfg.w_action_smooth * Jsm

    Jp = r_progress(s_pos, road_length)
    progress_reward = cfg.w_progress * Jp

    J_bumps = 0.0
    smooth = abs(z_B_ddot) <= cfg.a_B_comfort
    while bumps_passed < len(bump_ends) and s_pos >= bump_ends[bumps_passed]:
        bumps_passed += 1
        if smooth:
            J_bumps += cfg.w_bump_cross

    scale = float(np.clip(v / cfg.v_max, 0.0, 1.0))
    total = scale * core + tracking_penalty + jerk_smooth_penalty + progress_reward + J_bumps + cfg.Q_step
    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))

    bd = {
        "J_heave":    float(np.nan_to_num(Jh,  nan=0.0)),
        "J_wheel":    float(np.nan_to_num(Jw,  nan=0.0)),
        "J_long":     float(np.nan_to_num(Jl,  nan=0.0)),
        "J_speed":    float(np.nan_to_num(Js,  nan=0.0)),
        "J_jerk":     float(np.nan_to_num(Jj,  nan=0.0)),
        "J_smooth":   float(np.nan_to_num(Jsm, nan=0.0)),
        "J_progress": float(np.nan_to_num(Jp,  nan=0.0)),
        "J_bumps":    J_bumps,
        "J_total":    total,
    }
    return total, bd, bumps_passed

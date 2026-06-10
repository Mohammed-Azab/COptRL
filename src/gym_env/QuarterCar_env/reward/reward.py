import numpy as np

from QuarterCar_env.config.reward_params import RewardConfig


def r_progress(s_pos: float, road_length: float) -> float:
    if road_length <= 0.0:
        return 0.0
    return float(np.clip(s_pos / road_length, 0.0, 1.0))


def r_speed_band(v: float, v_min: float, v_ref: float) -> float:
    if v < v_min:
        return -1.0 - ((v_min - v) / v_min) ** 2
    # Mandl-style absolute tracking: symmetric penalty, zero only at v == v_ref
    return -abs(v_ref - v) / max(v_ref, 0.1)


def r_accel(a: float, a_comfort: float, accel_clip: float) -> float:
    a_c = float(np.clip(a, -accel_clip, accel_clip))
    return -(a_c / a_comfort) ** 2


def r_jerk(jerk: float, j_max: float, jerk_clip: float) -> float:
    j_c = float(np.clip(jerk, -jerk_clip, jerk_clip))
    return -(j_c / j_max) ** 2


def r_action_smooth(u_t: float, u_prev: float) -> float:
    return -(u_t - u_prev) ** 2


def r_heave(z_B_ddot: float, a_B_comfort: float, heave_clip: float) -> float:
    z_c = float(np.clip(z_B_ddot, -heave_clip, heave_clip))
    return -(z_c / a_B_comfort) ** 2


def r_wheel(z_W_ddot: float, a_W_comfort: float, wheel_clip: float) -> float:
    z_c = float(np.clip(z_W_ddot, -wheel_clip, wheel_clip))
    return -(z_c / a_W_comfort) ** 2


def compute_terminal_bonus(rms_accel: float, mean_speed: float, cfg: RewardConfig) -> float:
    # both conditions required
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

    rh = r_heave(z_B_ddot, cfg.a_B_comfort, cfg.reward_heave_clip)
    rw = r_wheel(z_W_ddot, cfg.a_W_comfort, cfg.reward_wheel_clip)
    ra = r_accel(filtered_a, cfg.a_comfort, cfg.reward_accel_clip)
    core = cfg.w_heave * rh + cfg.w_wheel * rw + cfg.w_accel * ra

    _v_ref = v_ref if v_ref > 0.0 else cfg.v_max
    rt = r_speed_band(v, cfg.v_min, _v_ref)
    # not velocity-scaled, see r_speed_band
    tracking_penalty = cfg.w_tracking * rt

    rj = r_jerk(filtered_jerk, cfg.j_max, cfg.reward_jerk_clip)
    rs = r_action_smooth(action, prev_action)
    # jerk and action_smooth not velocity-scaled
    jerk_smooth_penalty = cfg.w_jerk * rj + cfg.w_action_smooth * rs

    rp = r_progress(s_pos, road_length)
    progress_reward = cfg.w_progress * rp

    r_bumps = 0.0
    smooth = abs(z_B_ddot) <= cfg.a_B_comfort
    while bumps_passed < len(bump_ends) and s_pos >= bump_ends[bumps_passed]:
        bumps_passed += 1
        if smooth:
            r_bumps += cfg.w_bump_cross

    scale = float(np.clip(v / cfg.v_max, 0.0, 1.0))
    total = scale * core + tracking_penalty + jerk_smooth_penalty + progress_reward + r_bumps + cfg.step_bonus

    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))

    bd = {
        "r_heave":         float(np.nan_to_num(rh, nan=0.0)),
        "r_wheel":         float(np.nan_to_num(rw, nan=0.0)),
        "r_accel":         float(np.nan_to_num(ra, nan=0.0)),
        "r_tracking":      float(np.nan_to_num(rt, nan=0.0)),
        "r_jerk":          float(np.nan_to_num(rj, nan=0.0)),
        "r_action_smooth": float(np.nan_to_num(rs, nan=0.0)),
        "r_progress":      float(np.nan_to_num(rp, nan=0.0)),
        "r_bumps":         r_bumps,
        "reward_total":    total,
    }
    return total, bd, bumps_passed

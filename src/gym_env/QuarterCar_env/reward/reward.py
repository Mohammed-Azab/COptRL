import numpy as np

from QuarterCar_env.config.reward_params import RewardConfig

# comfort
def j_heave(z_B_ddot: float, g: float, heave_clip: float) -> float:
    z_c = float(np.clip(z_B_ddot, -heave_clip, heave_clip))
    return -(z_c / g) ** 2


def j_wheel(z_W_ddot: float, g: float, wheel_clip: float) -> float:
    z_c = float(np.clip(z_W_ddot, -wheel_clip, wheel_clip))
    return -(z_c / g) ** 2


def j_long(a: float, g: float, accel_clip: float) -> float:
    a_c = float(np.clip(a, -accel_clip, accel_clip))
    return -(a_c / g) ** 2


# Smoothness
def j_jerk(jerk: float, j_max: float, jerk_clip: float) -> float:
    j_c = float(np.clip(jerk, -jerk_clip, jerk_clip))
    return -(j_c / j_max) ** 2


def j_action_smooth(u_t: float, u_prev: float) -> float:
    return -(u_t - u_prev) ** 2


# Speed
def j_speed(v: float, v_min: float, v_init: float, v_limit: float = 1.0) -> float:
    if v < v_min:
        return -1.0 - ((v_min - v) / v_min) ** 2
    return -abs(v_init - v) / max(v_limit, 0.1)


# Terminal Rewards
def j_destination(s_pos: float, road_length: float, cfg: RewardConfig) -> float:
    return cfg.terminal_bonus if s_pos >= road_length - 1.0 else cfg.terminal_penalty


def j_time(t: float, t_max: float, cfg: RewardConfig) -> float:
    frac_used = float(np.clip(t / max(t_max, 1e-6), 0.0, 1.0))
    return cfg.Q_t * cfg.terminal_bonus * (1.0 - frac_used)


def compute_terminal_reward(t: float, t_max: float, road_length: float,
                             s_pos: float, cfg: RewardConfig) -> float:
    Jd = j_destination(s_pos, road_length, cfg)
    Jt = j_time(t, t_max, cfg) if Jd > 0 else 0.0
    return Jd + Jt


def compute_reward(
    v: float,
    z_B_ddot: float,
    z_W_ddot: float,
    filtered_a: float,
    filtered_jerk: float,
    prev_action: float,
    action: float,
    cfg: RewardConfig,
    v_init: float = 0.0,
) -> tuple[float, dict]:

    _v_init = v_init if v_init > 0.0 else cfg.v_limit

    Jh = j_heave(z_B_ddot, cfg.g, cfg.reward_heave_clip)
    Jw = j_wheel(z_W_ddot, cfg.g, cfg.reward_wheel_clip)
    Jl = j_long(filtered_a, cfg.g, cfg.reward_accel_clip)
    J_comfort = cfg.Q_zBddot * Jh + cfg.Q_zWddot * Jw + cfg.Q_a * Jl

    Js = j_speed(v, cfg.v_min, _v_init, v_limit=cfg.v_limit)
    J_speed = cfg.Q_v * Js

    Jj = j_jerk(filtered_jerk, cfg.j_max, cfg.reward_jerk_clip)
    Jas = j_action_smooth(action, prev_action)
    J_jerk = cfg.w_jerk * Jj + cfg.w_action_smooth * Jas

    scale = float(np.clip(v / _v_init, 0.0, 1.0))
    scale = 1.0

    total = scale * J_comfort + J_speed + J_jerk + cfg.Q_step
    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))

    bd = {
        "J_heave":    float(np.nan_to_num(Jh,  nan=0.0)),
        "J_wheel":    float(np.nan_to_num(Jw,  nan=0.0)),
        "J_long":     float(np.nan_to_num(Jl,  nan=0.0)),
        "J_speed":    float(np.nan_to_num(Js,  nan=0.0)),
        "J_jerk":     float(np.nan_to_num(Jj,  nan=0.0)),
        "J_smooth":   float(np.nan_to_num(Jas, nan=0.0)),
        "J_total":    total,
    }
    return total, bd

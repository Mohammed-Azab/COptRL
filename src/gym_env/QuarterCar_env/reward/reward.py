import numpy as np

from QuarterCar_env.config.reward_params import RewardConfig


def r_progress(v: float, v_max: float) -> float:
    return float(np.clip(v / v_max, 0.0, 1.0))


def r_speed_band(v: float, v_min: float, v_upper: float) -> float:
    # Always penalise distance from v_upper — no dead band (follows Mandl 2021 Eq. 4.21b).
    # A dead band [v_min, v_upper] = 0 lets the agent creep just above v_min for free.
    if v < v_min:
        # Extra penalty below minimum (near-stop adds -1 on top of the normal deviation)
        return -1.0 - ((v_min - v) / v_min) ** 2
    return -((v_upper - v) / v_upper) ** 2


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
    # require BOTH low body accel AND a minimum mean speed — prevents stop-and-wait exploit
    if rms_accel < cfg.a_limit and mean_speed >= cfg.v_min:
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
        rh = r_heave(z_B_ddot, cfg.a_B_comfort, cfg.reward_heave_clip)
        bd["r_heave"] = rh
        core += cfg.w_heave * rh
    else:
        bd["r_heave"] = 0.0

    if cfg.enable_wheel:
        rw = r_wheel(z_W_ddot, cfg.a_W_comfort, cfg.reward_wheel_clip)
        bd["r_wheel"] = rw
        core += cfg.w_wheel * rw
    else:
        bd["r_wheel"] = 0.0

    # r_tracking is intentionally excluded from velocity scaling:
    # at v≈0 a scaled tracking penalty vanishes, letting the agent learn stop-and-wait.
    tracking_penalty = 0.0
    if cfg.enable_tracking:
        rt = r_speed_band(v, cfg.v_min, v_upper)
        bd["r_tracking"] = rt
        tracking_penalty = cfg.w_tracking * rt
    else:
        bd["r_tracking"] = 0.0

    if cfg.enable_accel:
        ra = r_accel(filtered_a, cfg.a_comfort, cfg.reward_accel_clip)
        bd["r_accel"] = ra
        core += cfg.w_accel * ra
    else:
        bd["r_accel"] = 0.0

    # jerk and action_smooth are intentionally NOT velocity-scaled:
    # they measure self-induced longitudinal oscillation which the agent fully controls
    # and which should cost the same at any speed. Scaling them down at low speed lets
    # the agent oscillate freely while driving slowly — see TRIAL_ERROR.md Issue 5.
    jerk_smooth_penalty = 0.0
    if cfg.enable_jerk:
        rj = r_jerk(filtered_jerk, cfg.j_max, cfg.reward_jerk_clip)
        bd["r_jerk"] = rj
        jerk_smooth_penalty += cfg.w_jerk * rj
    else:
        bd["r_jerk"] = 0.0

    if cfg.enable_action_smooth:
        rs = r_action_smooth(action, prev_action)
        bd["r_action_smooth"] = rs
        jerk_smooth_penalty += cfg.w_action_smooth * rs
    else:
        bd["r_action_smooth"] = 0.0

    # progress: positive reward for forward movement — unscaled, always on
    progress_reward = 0.0
    if cfg.enable_progress:
        rp = r_progress(v, cfg.v_max)
        bd["r_progress"] = rp
        progress_reward = cfg.w_progress * rp
    else:
        bd["r_progress"] = 0.0

    scale = float(np.clip(v / cfg.v_max, 0.0, 1.0)) if cfg.enable_vel_scaling else 1.0
    total = scale * core + tracking_penalty + jerk_smooth_penalty + progress_reward

    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))
    for key in bd:
        bd[key] = float(np.nan_to_num(bd[key], nan=0.0, posinf=0.0, neginf=0.0))

    bd["r_curve"]      = 0.0
    bd["r_bumps"]      = 0.0   # filled by env when a bump end is cleared
    bd["reward_total"] = total
    return total, bd

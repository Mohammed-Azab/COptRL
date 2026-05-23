from QuarterCar_env.config.reward_params import RewardConfig

def reward_bounds(cfg: RewardConfig, n_steps: int) -> dict:
    """
    Theoretical per-step and episode reward bounds for a given RewardConfig.

    Per-step max: w_comfort_bonus × 1.0 when accel == 0 and all penalties are zero.
    Per-step min: all weighted penalty terms at their reward clip boundaries.

    The episode_min is a hard mathematical limit -> in practice the IIR filters


    well above episode_min (typical random-agent range is roughly -500 to 0).
    """
    per_step_max = 0.0
    per_step_min = 0.0

    if cfg.enable_comfort_bonus:
        # best: filtered_a == 0  → r_comfort_bonus = 1
        per_step_max += cfg.w_comfort_bonus * 1.0

    if cfg.enable_tracking:
        # worst: v == 0 (full stop)  → r_speed_band = -(v_min/v_min)² = -1
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
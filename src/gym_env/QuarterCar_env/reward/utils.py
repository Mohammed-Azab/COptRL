from QuarterCar_env.config.reward_params import RewardConfig


def reward_bounds(cfg: RewardConfig, n_steps: int) -> dict:
    """Theoretical per-step and episode reward bounds.

    Velocity-scaled terms (heave, wheel, accel, jerk, smooth) are worst at v=v_max (scale=1).
    r_tracking is unscaled and worst at v=0 (-2.0), but = 0 at v=v_max.
    Both cannot be simultaneously worst, so the tighter bound is the velocity-scaled scenario.
    """
    # worst case for velocity-scaled comfort terms occurs at v=v_max (scale=1).
    # At v=v_max, r_tracking=0, so tracking does not contribute to the minimum.
    # (At v=0, scale=0 so comfort terms vanish — the two worst-cases are mutually exclusive.)
    per_step_min = 0.0
    if cfg.enable_heave:
        per_step_min += cfg.w_heave * -(cfg.reward_heave_clip / cfg.a_B_comfort) ** 2
    if cfg.enable_wheel:
        per_step_min += cfg.w_wheel * -(cfg.reward_wheel_clip / cfg.a_W_comfort) ** 2
    if cfg.enable_accel:
        per_step_min += cfg.w_accel * -(cfg.reward_accel_clip / cfg.a_comfort) ** 2
    if cfg.enable_jerk:
        per_step_min += cfg.w_jerk * -(cfg.reward_jerk_clip / cfg.j_max) ** 2
    if cfg.enable_action_smooth:
        per_step_min += cfg.w_action_smooth * (-4.0)

    per_step_max = 0.0

    return {
        "per_step_max": round(per_step_max, 4),
        "per_step_min": round(per_step_min, 4),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus,   1),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 1),
        "n_steps": n_steps,
    }

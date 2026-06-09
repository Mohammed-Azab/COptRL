from QuarterCar_env.config.reward_params import RewardConfig


def reward_bounds(cfg: RewardConfig, n_steps: int) -> dict:
    # theoretical per-step and episode reward bounds.
    # velocity-scaled terms (heave, wheel, accel, jerk, smooth) hit their worst
    # at v=v_max (scale=1). r_tracking is worst at v=0 but zero at v=v_max, so
    # the two worst cases are mutually exclusive — we use the velocity-scaled one.
    # worst case for velocity-scaled terms: v=v_max, tracking=0
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

    # progress reward: max is w_progress × 1.0 (at v=v_max), unscaled
    per_step_max = cfg.w_progress if cfg.enable_progress else 0.0

    return {
        "per_step_max": round(per_step_max, 4),
        "per_step_min": round(per_step_min, 4),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus,   1),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 1),
        "n_steps": n_steps,
    }

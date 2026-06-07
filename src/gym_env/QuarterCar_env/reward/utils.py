from QuarterCar_env.config.reward_params import RewardConfig


def reward_bounds(cfg: RewardConfig, n_steps: int) -> dict:
    """Theoretical per-step reward bounds. Velocity scaling (v/v_max) is excluded."""
    per_step_max = 0.0
    per_step_min = 0.0

    if cfg.enable_heave:
        per_step_min += cfg.w_heave * (-1.0)

    if cfg.enable_wheel:
        per_step_min += cfg.w_wheel * (-1.0)

    if cfg.enable_tracking:
        per_step_min += cfg.w_tracking * (-1.0)

    if cfg.enable_accel:
        per_step_min += cfg.w_accel * -(cfg.reward_accel_clip / cfg.a_comfort) ** 2

    if cfg.enable_jerk:
        per_step_min += cfg.w_jerk * -(cfg.reward_jerk_clip / cfg.j_max) ** 2

    if cfg.enable_action_smooth:
        per_step_min += cfg.w_action_smooth * (-4.0)

    return {
        "per_step_max": round(per_step_max, 6),
        "per_step_min": round(per_step_min, 6),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus,   4),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 4),
        "n_steps": n_steps,
    }

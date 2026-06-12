from QuarterCar_env.config.reward_params import RewardConfig

_J_SMOOTH_WORST   = -(2.0 ** 2)  # r_action_smooth at max diff (±1 → 2)
_J_SPEED_WORST    = -(1.0 + 1.0) # j_speed at v=0: -1.0 - (v_min/v_min)² = -2.0


def reward_bounds(cfg: RewardConfig, n_steps: int, n_bumps: int = 0) -> dict:
    per_step_max = -cfg.Q_step  # best step: zero comfort/speed/jerk penalty

    per_step_min = (
        cfg.Q_zBddot * -(cfg.reward_heave_clip / cfg.g) ** 2
        + cfg.Q_zWddot * -(cfg.reward_wheel_clip / cfg.g) ** 2
        + cfg.Q_a      * -(cfg.reward_accel_clip / cfg.g) ** 2
        + cfg.w_jerk   * -(cfg.reward_jerk_clip  / cfg.j_max) ** 2
        + cfg.w_action_smooth * _J_SMOOTH_WORST
        + cfg.Q_v             * _J_SPEED_WORST
        - cfg.Q_step
    )

    return {
        "per_step_max": round(per_step_max, 4),
        "per_step_min": round(per_step_min, 4),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus, 1),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 1),
        "n_steps":  n_steps,
        "n_bumps":  n_bumps,
    }

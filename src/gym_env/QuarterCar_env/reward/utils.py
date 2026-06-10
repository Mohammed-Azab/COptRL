from QuarterCar_env.config.reward_params import RewardConfig


def reward_bounds(cfg: RewardConfig, n_steps: int) -> dict:
    # worst-case per-step reward: velocity-scaled terms at v=v_max (scale=1)
    # tracking worst case: v=v_min → r_speed_band = -((v_limit-v_min)/v_limit)²
    per_step_min = (
        cfg.w_heave   * -(cfg.reward_heave_clip / cfg.a_B_comfort) ** 2
        + cfg.w_wheel * -(cfg.reward_wheel_clip / cfg.a_W_comfort) ** 2
        + cfg.w_accel * -(cfg.reward_accel_clip / cfg.a_comfort)   ** 2
        + cfg.w_jerk  * -(cfg.reward_jerk_clip  / cfg.j_max)       ** 2
        + cfg.w_action_smooth * (-4.0)
        + cfg.w_tracking * -((cfg.v_limit - cfg.v_min) / cfg.v_limit) ** 2
        + cfg.step_bonus
    )
    per_step_max = cfg.w_progress + cfg.step_bonus

    return {
        "per_step_max": round(per_step_max, 4),
        "per_step_min": round(per_step_min, 4),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus,   1),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 1),
        "n_steps": n_steps,
    }

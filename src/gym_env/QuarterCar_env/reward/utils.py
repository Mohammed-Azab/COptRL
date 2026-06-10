from QuarterCar_env.config.reward_params import RewardConfig

# Worst-case constants derived from formula bounds (not config-dependent)
_R_SMOOTH_WORST   = -(2.0 ** 2)   # r_action_smooth: action ∈ [-1,1], max diff = 2 → -(2)²
_R_TRACKING_WORST = -(1.0 + 1.0)  # r_speed_band at v=0: -1.0 - (v_min/v_min)²  = -2.0


def reward_bounds(cfg: RewardConfig, n_steps: int, n_bumps: int = 0) -> dict:
    # Theoretical per-step bounds: each term evaluated at its independent worst case.
    # Comfort terms: worst at v=v_max (velocity scale = 1).
    # Tracking:      worst at v=0 (r_speed_band = -2.0).
    # Jerk/smooth:   not velocity-scaled, evaluated at clip/saturation limit.
    # These extremes cannot all occur in the same step; the combined bound is
    # therefore more conservative (more negative) than any single real step.
    #
    # n_bumps: number of bump crossings expected in the episode.
    # Each crossing fires a one-time +w_bump_cross reward (not per-step).
    # Pass 0 to get a per-step-only bound (correct for flat-road episodes).
    per_step_min = (
        cfg.w_heave   * -(cfg.reward_heave_clip / cfg.a_B_comfort) ** 2
        + cfg.w_wheel * -(cfg.reward_wheel_clip / cfg.a_W_comfort) ** 2
        + cfg.w_accel * -(cfg.reward_accel_clip / cfg.a_comfort)   ** 2
        + cfg.w_jerk  * -(cfg.reward_jerk_clip  / cfg.j_max)       ** 2
        + cfg.w_action_smooth * _R_SMOOTH_WORST
        + cfg.w_tracking      * _R_TRACKING_WORST
        + cfg.step_bonus
    )
    per_step_max  = cfg.w_progress + cfg.step_bonus
    crossing_max  = cfg.w_bump_cross * n_bumps

    return {
        "per_step_max": round(per_step_max, 4),
        "per_step_min": round(per_step_min, 4),
        "episode_max":  round(per_step_max * n_steps + crossing_max + cfg.terminal_bonus,   1),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 1),
        "n_steps":  n_steps,
        "n_bumps":  n_bumps,
    }

"""
Comfort-aware reward system for the QuarterCar speed-planning environment.

Reward composition:
    R = w_comfort_bonus * r_comfort_bonus  -> positive per-step reward for smooth riding
      + w_tracking      * r_speed_band     -> stay within [v_min, v_max]
      + w_accel         * r_accel          -> longitudinal acceleration penalty
      + w_jerk          * r_jerk           -> jerk penalty
      + w_action_smooth * r_action_smooth  -> discontinuous command penalty

    func r_speed_band():
        Returns 0 inside [v_min, v_max]; penalises stopping or speeding.
        Returns -1 at v = 0 (full stop) and -1 when v == 2 × v_max.

    func r_accel():
        Returns -(filtered_a / a_comfort)². Returns -1 when |a| == a_comfort.

    func r_jerk():
        Returns -(filtered_jerk / j_max)². Returns -1 when |jerk| == j_max.

    func r_action_smooth():
        Returns -(u_t - u_{t-1})². Penalises sudden command changes.

    func r_comfort_bonus():
        Returns max(0, 1 - (filtered_a / a_comfort)²). Positive only inside comfort band.

    func compute_reward():
        Returns (total_reward, breakdown_dict).
        breakdown_dict keys: r_tracking, r_comfort_bonus, r_accel, r_jerk,
                             r_action_smooth, r_curve (always 0), reward_total.
"""

from QuarterCar_env.reward.reward import (
    r_speed_band,
    r_accel,
    r_jerk,
    r_action_smooth,
    r_comfort_bonus,
    compute_reward,
    compute_terminal_bonus,
)

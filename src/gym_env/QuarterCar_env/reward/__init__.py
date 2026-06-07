"""
Reward system for the QuarterCar speed-planning environment.

Reward composition:
    R_step = (v / v_max) * (
        w_heave * r_heave + w_wheel * r_wheel
        + w_tracking * r_tracking + w_accel * r_accel
        + w_jerk * r_jerk + w_action_smooth * r_action_smooth
    )
"""

from QuarterCar_env.reward.reward import (
    r_speed_band,
    r_accel,
    r_jerk,
    r_action_smooth,
    r_heave,
    r_wheel,
    compute_reward,
    compute_terminal_bonus,
)

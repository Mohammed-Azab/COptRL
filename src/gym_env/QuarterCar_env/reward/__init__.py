"""
Comfort-aware reward system for the QuarterCar speed-planning environment.

Reward composition:
    R = w_comfort_bonus * r_comfort_bonus  -> positive per-step reward for smooth riding
      + w_tracking      * r_speed_band     -> stay within [v_min, v_upper];
      + w_accel         * r_accel          -> longitudinal acceleration (ISO 2631)
      + w_jerk          * r_jerk           -> rapid acceleration changes
      + w_action_smooth * r_action_smooth  -> discontinuous commands
      + w_curve         * r_curve          -> lateral discomfort from road curvature

Step reward range with default config:
    per-step  ∈ [-4.8, +0.8]
    episode   ∈ [-1300, +300]   (theoretical; practical bad-agent range ≈ -400 to 0)

Episode max (+300): perfect comfort bonus every step + terminal bonus.
"""

from QuarterCar_env.reward.reward import (
    r_speed_band,
    r_accel,
    r_jerk,
    r_action_smooth,
    r_curve,
    compute_reward,
    compute_terminal_bonus,
)


"""
notes:

    func r_speed_band():
        -> Returns 0 anywhere inside [v_min, v_upper] -> the agent is free to choose in between
        -> Returns -1 at v = 0 (full stop).
    
    func r_accel():
        -> Longitudinal accelration 
        -> Returns -1 when |a| == a_comfort.

    func r_jerk():
        -> Jerk penalty.
        -> Returns -1 when |jerk| == j_max.

    func r_action_smooth():
        Penalises sudden command changes. 
        Returns 0 when unchanged.

    func r_curve():
        -> Lateral penalty from road curvature.
        -> a_lat = v^2 x |curvature|.

    func r_comfort_bonus():
        -> Positive per-step reward for riding inside the comfort region 
        -> it encourge the agent to stay in the comfort region 
        
    func compute_reward()
        -> Returns (total_reward, breakdown)
        -> Breakdown is a flat dict with every term value plus "reward_total".

        -> Args:
            v:             Current speed [m/s].
            v_upper:       Upper band limit [m/s] -> v_max in constant mode, curve-adjusted otherwise.
            a_actual:      Raw finite-difference acceleration [m/s²].
            filtered_a:    IIR-smoothed acceleration [m/s²] -> used for r_accel.
            jerk:          Raw finite-difference jerk [m/s³].
            filtered_jerk: IIR-smoothed jerk [m/s³] -> used for r_jerk.
            prev_action:   Previous normalised action in [-1, 1].
            action:        Current normalised action in [-1, 1].
            curvature:     Road curvature [m^-1].
            cfg:           RewardConfig.

"""
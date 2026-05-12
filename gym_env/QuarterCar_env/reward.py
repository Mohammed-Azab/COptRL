"""Reward function for quarter-car active suspension RL."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    w_comfort:     float = 1.0
    w_travel:      float = 0.3
    w_tyre:        float = 0.3
    w_effort:      float = 0.1
    a_limit:       float = 10.0     # m/s²
    travel_limit:  float = 0.08     # m
    tyre_limit:    float = 0.04     # m
    terminal_bonus: float = 10.0
    F_max:         float = 10000.0  # N
    # Speed-control terms (only used when control_mode includes speed)
    w_time:       float = 0.5    # [1/step] — flat per-step time penalty (×DT externally)
    w_speed_err:  float = 0.3    # [dimensionless] — penalty for (v − v_ref)² / v_ref²
    v_ref:        float = 10.0   # m/s — reference speed, overridden per step by the env


def compute_reward(
    z_s_ddot: float,
    suspension_travel: float,
    tyre_deflection: float,
    F_act: float,
    config: RewardConfig,
    v: Optional[float] = None,   # current longitudinal speed; None → skip speed terms
) -> float:
    r_comfort = -config.w_comfort * (z_s_ddot         / config.a_limit)      ** 2
    r_travel  = -config.w_travel  * (suspension_travel / config.travel_limit) ** 2
    r_tyre    = -config.w_tyre    * (tyre_deflection   / config.tyre_limit)   ** 2
    r_effort  = -config.w_effort  * (F_act             / config.F_max)        ** 2
    total = r_comfort + r_travel + r_tyre + r_effort

    if v is not None:
        r_time      = -config.w_time                                                    # flat step penalty
        v_ref_safe  = config.v_ref if abs(config.v_ref) > 1e-6 else 1e-6
        r_speed_err = -config.w_speed_err * ((v - config.v_ref) / v_ref_safe) ** 2     # normalised speed error
        total += r_time + r_speed_err

    return total


def compute_terminal_bonus(rms_accel: float, config: RewardConfig) -> float:
    return config.terminal_bonus * (1.0 - min(rms_accel / config.a_limit, 1.0))

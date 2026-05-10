"""Reward function for quarter-car active suspension RL."""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_comfort:     float = 1.0
    w_travel:      float = 0.3
    w_tyre:        float = 0.3
    w_effort:      float = 0.1
    a_limit:       float = 10.0     # m/s^2
    travel_limit:  float = 0.08     # m
    tyre_limit:    float = 0.04     # m
    terminal_bonus: float = 10.0
    F_max:         float = 10000.0  # N


def compute_reward(
    z_s_ddot: float,
    suspension_travel: float,
    tyre_deflection: float,
    F_act: float,
    config: RewardConfig,
) -> float:
    r_comfort = -config.w_comfort * (z_s_ddot         / config.a_limit)      ** 2
    r_travel  = -config.w_travel  * (suspension_travel / config.travel_limit) ** 2
    r_tyre    = -config.w_tyre    * (tyre_deflection   / config.tyre_limit)   ** 2
    r_effort  = -config.w_effort  * (F_act             / config.F_max)        ** 2
    return r_comfort + r_travel + r_tyre + r_effort


def compute_terminal_bonus(rms_accel: float, config: RewardConfig) -> float:
    return config.terminal_bonus * (1.0 - min(rms_accel / config.a_limit, 1.0))

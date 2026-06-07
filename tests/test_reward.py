import numpy as np
import sys
sys.path.insert(0, "src/road")
from QuarterCar_env.config.reward_params import RewardConfig
from QuarterCar_env.reward.reward import compute_reward, r_heave, r_wheel

cfg = RewardConfig()


def test_r_heave_zero_at_zero():
    assert r_heave(0.0, cfg.a_B_comfort) == 0.0


def test_r_heave_negative_for_nonzero():
    assert r_heave(5.0, cfg.a_B_comfort) < 0.0


def test_r_wheel_negative_for_nonzero():
    assert r_wheel(10.0, cfg.a_W_comfort) < 0.0


def test_compute_reward_has_heave_key():
    total, bd = compute_reward(
        v=10.0, v_upper=15.0,
        z_B_ddot=2.0, z_W_ddot=5.0,
        filtered_a=0.5, filtered_jerk=1.0,
        prev_action=0.0, action=0.1,
        cfg=cfg,
    )
    assert "r_heave" in bd
    assert "r_wheel" in bd


def test_velocity_scaling_reduces_reward():
    total_zero, _ = compute_reward(
        v=0.0, v_upper=15.0,
        z_B_ddot=2.0, z_W_ddot=5.0,
        filtered_a=0.5, filtered_jerk=1.0,
        prev_action=0.0, action=0.1,
        cfg=cfg,
    )
    total_full, _ = compute_reward(
        v=cfg.v_max, v_upper=15.0,
        z_B_ddot=2.0, z_W_ddot=5.0,
        filtered_a=0.5, filtered_jerk=1.0,
        prev_action=0.0, action=0.1,
        cfg=cfg,
    )
    assert abs(total_zero) < abs(total_full)


def test_no_comfort_bonus_key():
    _, bd = compute_reward(
        v=10.0, v_upper=15.0,
        z_B_ddot=0.0, z_W_ddot=0.0,
        filtered_a=0.0, filtered_jerk=0.0,
        prev_action=0.0, action=0.0,
        cfg=cfg,
    )
    assert "r_comfort_bonus" not in bd

import numpy as np
import sys
sys.path.insert(0, "src/road")
from QuarterCar_env.config.reward_params import RewardConfig
from QuarterCar_env.reward.reward import compute_reward, j_heave, j_wheel

cfg = RewardConfig()


def test_j_heave_zero_at_zero():
    assert j_heave(0.0, cfg.g, cfg.reward_heave_clip) == 0.0


def test_j_heave_negative_for_nonzero():
    assert j_heave(0.5, cfg.g, cfg.reward_heave_clip) < 0.0


def test_j_heave_capped_at_clip():
    at_clip = j_heave(cfg.reward_heave_clip,      cfg.g, cfg.reward_heave_clip)
    above   = j_heave(cfg.reward_heave_clip * 10, cfg.g, cfg.reward_heave_clip)
    assert at_clip == above


def test_j_wheel_negative_for_nonzero():
    assert j_wheel(10.0, cfg.g, cfg.reward_wheel_clip) < 0.0


def test_compute_reward_has_mandl_keys():
    total, bd, _ = compute_reward(
        v=10.0,
        z_B_ddot=2.0, z_W_ddot=5.0,
        filtered_a=0.5, filtered_jerk=1.0,
        prev_action=0.0, action=0.1,
        cfg=cfg, s_pos=50.0, road_length=100.0,
    )
    assert "J_heave" in bd
    assert "J_wheel" in bd
    assert "J_speed" in bd
    assert "J_long"  in bd


def test_velocity_scaling_reduces_reward():
    total_zero, _, _ = compute_reward(
        v=0.0,
        z_B_ddot=2.0, z_W_ddot=5.0,
        filtered_a=0.5, filtered_jerk=1.0,
        prev_action=0.0, action=0.1,
        cfg=cfg, s_pos=0.0, road_length=100.0,
    )
    total_full, _, _ = compute_reward(
        v=cfg.v_max,
        z_B_ddot=2.0, z_W_ddot=5.0,
        filtered_a=0.5, filtered_jerk=1.0,
        prev_action=0.0, action=0.1,
        cfg=cfg, s_pos=100.0, road_length=100.0,
    )
    assert abs(total_zero) < abs(total_full)


def test_no_comfort_bonus_key():
    _, bd, _ = compute_reward(
        v=10.0,
        z_B_ddot=0.0, z_W_ddot=0.0,
        filtered_a=0.0, filtered_jerk=0.0,
        prev_action=0.0, action=0.0,
        cfg=cfg, s_pos=50.0, road_length=100.0,
    )
    assert "r_comfort_bonus" not in bd

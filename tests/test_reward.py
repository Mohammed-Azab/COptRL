import pytest
import numpy as np
from QuarterCar_env.reward.reward import (
    r_tracking,
    r_accel,
    r_jerk,
    r_action_smooth,
    r_curve,
    r_energy,
)


# --- r_tracking ---

def test_r_tracking_zero_error():
    assert r_tracking(5.0, 5.0, 20.0) == pytest.approx(0.0)


def test_r_tracking_full_error():
    # error = v_max → -(1)^2 = -1.0
    assert r_tracking(0.0, 20.0, 20.0) == pytest.approx(-1.0)


def test_r_tracking_half_error():
    # error = 10, v_max = 20 → -(0.5)^2 = -0.25
    assert r_tracking(0.0, 10.0, 20.0) == pytest.approx(-0.25)


def test_r_tracking_overspeed():
    # v > v_target: still penalised — error = 5, v_max = 20 → -(5/20)^2 = -0.0625
    assert r_tracking(25.0, 20.0, 20.0) == pytest.approx(-0.0625)


# --- r_accel ---

def test_r_accel_zero():
    assert r_accel(0.0, 3.0, 15.0) == pytest.approx(0.0)


def test_r_accel_at_comfort_threshold():
    # a = a_comfort → -(1)^2 = -1.0
    assert r_accel(3.0, 3.0, 15.0) == pytest.approx(-1.0)


def test_r_accel_clipping():
    # a > accel_clip → clipped, so result = -(accel_clip/a_comfort)^2
    assert r_accel(100.0, 3.0, 15.0) == pytest.approx(-(15.0 / 3.0) ** 2)


def test_r_accel_negative():
    # braking acceleration treated same as positive
    assert r_accel(-3.0, 3.0, 15.0) == pytest.approx(-1.0)


# --- r_jerk ---

def test_r_jerk_zero():
    assert r_jerk(0.0, 10.0, 50.0) == pytest.approx(0.0)


def test_r_jerk_at_max():
    assert r_jerk(10.0, 10.0, 50.0) == pytest.approx(-1.0)


def test_r_jerk_clipping():
    assert r_jerk(100.0, 10.0, 50.0) == pytest.approx(-(50.0 / 10.0) ** 2)


# --- r_action_smooth ---

def test_r_action_smooth_no_change():
    assert r_action_smooth(0.5, 0.5) == pytest.approx(0.0)


def test_r_action_smooth_max_change():
    # full range: from -1 to 1 → -(2)^2 = -4.0
    assert r_action_smooth(1.0, -1.0) == pytest.approx(-4.0)


def test_r_action_smooth_small_change():
    assert r_action_smooth(0.3, 0.0) == pytest.approx(-0.09)


# --- r_curve ---

def test_r_curve_zero_curvature():
    assert r_curve(10.0, 0.0, 4.0, 0.5) == pytest.approx(0.0)


def test_r_curve_at_limit():
    # a_lat = v^2 * k = 4^2 * 0.25 = 4.0 → -(4/4)^2 = -1.0
    assert r_curve(4.0, 0.25, 4.0, 0.5) == pytest.approx(-1.0)


def test_r_curve_clipping():
    # curvature = 2.0 > clip=0.5 → clipped to 0.5
    # a_lat = 4^2 * 0.5 = 8.0 → -(8/4)^2 = -4.0
    assert r_curve(4.0, 2.0, 4.0, 0.5) == pytest.approx(-4.0)


def test_r_curve_negative_curvature():
    # abs applied — same penalty as positive
    assert r_curve(4.0, -0.25, 4.0, 0.5) == pytest.approx(-1.0)


# --- r_energy ---

def test_r_energy_zero():
    assert r_energy(0.0) == pytest.approx(0.0)


def test_r_energy_full_positive():
    assert r_energy(1.0) == pytest.approx(-1.0)


def test_r_energy_full_negative():
    assert r_energy(-1.0) == pytest.approx(-1.0)


def test_r_energy_half():
    assert r_energy(0.5) == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# compute_v_target
# ---------------------------------------------------------------------------
from QuarterCar_env.reward.reward import compute_v_target
from QuarterCar_env.config.reward_params import RewardConfig


def test_compute_v_target_constant():
    cfg = RewardConfig()
    assert compute_v_target(15.0, "constant", 0.3, cfg) == pytest.approx(15.0)


def test_compute_v_target_curvature_aware_high_curvature():
    cfg = RewardConfig(a_lat_comfort=2.0, min_curve_speed=2.0, max_curve_speed=20.0)
    # high curvature → speed limit below v_ref=20
    result = compute_v_target(20.0, "curvature_aware", 0.5, cfg)
    assert result < 20.0
    assert result >= cfg.min_curve_speed


def test_compute_v_target_curvature_aware_zero_curvature():
    cfg = RewardConfig(a_lat_comfort=2.0, min_curve_speed=2.0, max_curve_speed=20.0)
    # zero curvature → sqrt(2/1e-6) >> v_ref → returns v_ref unchanged
    result = compute_v_target(15.0, "curvature_aware", 0.0, cfg)
    assert result == pytest.approx(15.0)


def test_compute_v_target_external_passthrough():
    cfg = RewardConfig()
    assert compute_v_target(12.0, "external", 0.1, cfg) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------
from QuarterCar_env.reward.reward import compute_reward


def _zero_cfg(**overrides):
    """RewardConfig with all weights zero except those in overrides."""
    base = dict(
        w_tracking=0.0, w_accel=0.0, w_jerk=0.0,
        w_action_smooth=0.0, w_curve=0.0, w_energy=0.0,
    )
    base.update(overrides)
    return RewardConfig(**base)


def test_compute_reward_returns_tuple():
    cfg = RewardConfig()
    result = compute_reward(10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    assert isinstance(result, tuple) and len(result) == 2
    reward, breakdown = result
    assert isinstance(reward, float)
    assert isinstance(breakdown, dict)


def test_compute_reward_breakdown_keys():
    cfg = RewardConfig()
    _, bd = compute_reward(10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    for key in ("r_tracking", "r_accel", "r_jerk", "r_action_smooth",
                "r_curve", "r_energy", "reward_total"):
        assert key in bd, f"Missing key: {key}"


def test_compute_reward_tracking_only():
    # only tracking weight active; v=0, v_target=20, v_max=20 → r_tracking=-1
    cfg = _zero_cfg(w_tracking=1.0, v_max=20.0)
    reward, bd = compute_reward(0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    assert reward == pytest.approx(-1.0)
    assert bd["r_tracking"] == pytest.approx(-1.0)


def test_compute_reward_disabled_term_contributes_zero():
    cfg = RewardConfig(enable_tracking=False, w_accel=0.0, w_jerk=0.0,
                       w_action_smooth=0.0, w_curve=0.0, w_energy=0.0)
    reward, bd = compute_reward(0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    assert reward == pytest.approx(0.0)
    assert bd["r_tracking"] == pytest.approx(0.0)


def test_compute_reward_no_nan():
    cfg = RewardConfig()
    reward, bd = compute_reward(float('nan'), 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    assert not np.isnan(reward)
    for v in bd.values():
        assert not np.isnan(v)


def test_compute_reward_total_in_breakdown():
    cfg = RewardConfig()
    reward, bd = compute_reward(10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    assert bd["reward_total"] == pytest.approx(reward)


# ---------------------------------------------------------------------------
# compute_terminal_bonus
# ---------------------------------------------------------------------------
from QuarterCar_env.reward.reward import compute_terminal_bonus


def test_terminal_bonus_when_rms_below_limit():
    cfg = RewardConfig(a_limit=10.0, terminal_bonus=100.0, terminal_penalty=-100.0)
    assert compute_terminal_bonus(5.0, cfg) == pytest.approx(100.0)


def test_terminal_penalty_when_rms_above_limit():
    cfg = RewardConfig(a_limit=10.0, terminal_bonus=100.0, terminal_penalty=-100.0)
    assert compute_terminal_bonus(15.0, cfg) == pytest.approx(-100.0)


def test_terminal_bonus_at_exact_limit():
    cfg = RewardConfig(a_limit=10.0, terminal_bonus=100.0, terminal_penalty=-100.0)
    # rms == a_limit: strictly less required for bonus → penalty
    assert compute_terminal_bonus(10.0, cfg) == pytest.approx(-100.0)

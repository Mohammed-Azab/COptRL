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

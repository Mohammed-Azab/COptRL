"""Tests for reward functions."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from quarter_car_core.reward import RewardConfig, compute_reward, compute_terminal_bonus


cfg = RewardConfig()


def test_reward_nonpositive_typical():
    r = compute_reward(5.0, 0.04, 0.02, 3000.0, cfg)
    assert r <= 0.0


def test_reward_zero_for_zero_inputs():
    r = compute_reward(0.0, 0.0, 0.0, 0.0, cfg)
    assert r == pytest.approx(0.0)


def test_larger_accel_more_negative():
    r_small = compute_reward(1.0, 0.0, 0.0, 0.0, cfg)
    r_large = compute_reward(8.0, 0.0, 0.0, 0.0, cfg)
    assert r_large < r_small


def test_larger_travel_more_negative():
    r_small = compute_reward(0.0, 0.01, 0.0, 0.0, cfg)
    r_large = compute_reward(0.0, 0.06, 0.0, 0.0, cfg)
    assert r_large < r_small


def test_effort_penalty_grows_with_force():
    r_zero  = compute_reward(0.0, 0.0, 0.0, 0.0, cfg)
    r_force = compute_reward(0.0, 0.0, 0.0, 5000.0, cfg)
    assert r_force < r_zero


def test_terminal_bonus_positive_good_run():
    bonus = compute_terminal_bonus(0.5, cfg)
    assert bonus > 0.0


def test_terminal_bonus_at_limit():
    bonus = compute_terminal_bonus(cfg.a_limit, cfg)
    assert bonus == pytest.approx(0.0, abs=1e-9)


def test_terminal_bonus_clamped_above_limit():
    bonus = compute_terminal_bonus(cfg.a_limit * 2, cfg)
    assert bonus == pytest.approx(0.0, abs=1e-9)


def test_all_weights_contribute():
    """Each term should contribute independently."""
    cfg2 = RewardConfig(w_comfort=1.0, w_travel=0.0, w_tyre=0.0, w_effort=0.0)
    r1 = compute_reward(5.0, 0.0, 0.0, 0.0, cfg2)
    cfg3 = RewardConfig(w_comfort=2.0, w_travel=0.0, w_tyre=0.0, w_effort=0.0)
    r2 = compute_reward(5.0, 0.0, 0.0, 0.0, cfg3)
    assert r2 == pytest.approx(2 * r1)

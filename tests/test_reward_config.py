import pytest
from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert cfg.w_tracking == pytest.approx(1.0)
    assert cfg.w_accel == pytest.approx(0.5)
    assert cfg.w_jerk == pytest.approx(0.3)
    assert cfg.w_action_smooth == pytest.approx(0.2)
    assert cfg.w_curve == pytest.approx(0.0)
    assert cfg.w_energy == pytest.approx(0.1)
    assert cfg.a_comfort == pytest.approx(3.0)
    assert cfg.j_max == pytest.approx(10.0)
    assert cfg.v_max == pytest.approx(20.0)
    assert cfg.a_max == pytest.approx(5.0)
    assert cfg.enable_tracking is True
    assert cfg.enable_curve is True
    assert cfg.obs_enable_accel is True
    assert cfg.obs_enable_curvature_preview is False


def test_load_reward_config_returns_reward_config():
    cfg = load_reward_config()
    assert isinstance(cfg, RewardConfig)


def test_load_reward_config_matches_yaml():
    cfg = load_reward_config()
    assert cfg.w_tracking == pytest.approx(1.0)
    assert cfg.target_speed_mode == "constant"

import pytest
import numpy as np
from QuarterCar_env.envs.quarter_car_env import QuarterCarEnv
from QuarterCar_env.config.reward_params import RewardConfig


@pytest.fixture
def env():
    e = QuarterCarEnv(road_profile='flat')
    yield e
    e.close()


@pytest.fixture
def env_no_extras():
    cfg = RewardConfig(
        obs_enable_accel=False,
        obs_enable_jerk=False,
        obs_enable_prev_action=False,
        obs_enable_curvature=False,
    )
    e = QuarterCarEnv(road_profile='flat', reward_config=cfg)
    yield e
    e.close()


# --- action space ---

def test_action_space_shape(env):
    assert env.action_space.shape == (1,)


def test_action_space_bounds(env):
    assert env.action_space.low[0] == pytest.approx(-1.0)
    assert env.action_space.high[0] == pytest.approx(1.0)


# --- observation space ---

def test_obs_space_default_size(env):
    # 8 base + 2 speed + 4 extras (accel, jerk, prev_action, curvature) = 14
    assert env.observation_space.shape == (14,)


def test_obs_space_minimal_size(env_no_extras):
    # 8 base + 2 speed = 10
    assert env_no_extras.observation_space.shape == (10,)


def test_obs_shape_matches_space_after_reset(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_obs_in_bounds_after_reset(env):
    obs, _ = env.reset()
    assert np.all(obs >= env.observation_space.low - 1e-4)
    assert np.all(obs <= env.observation_space.high + 1e-4)


# --- step ---

def test_step_returns_five_elements(env):
    env.reset()
    result = env.step(np.array([0.0], dtype=np.float32))
    assert len(result) == 5


def test_step_obs_shape(env):
    env.reset()
    obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert obs.shape == env.observation_space.shape


def test_step_reward_is_float(env):
    env.reset()
    _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert isinstance(reward, float)


def test_step_obs_in_bounds(env):
    env.reset()
    obs, _, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
    assert np.all(obs >= env.observation_space.low - 1e-4)
    assert np.all(obs <= env.observation_space.high + 1e-4)


# --- reward breakdown in info ---

def test_reward_breakdown_in_info(env):
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    for key in ("r_tracking", "r_accel", "r_jerk", "r_action_smooth",
                "r_curve", "r_energy", "reward_total"):
        assert key in info, f"Missing key in info: {key}"


# --- speed dynamics ---

def test_positive_action_increases_speed(env):
    env.reset()
    env._v = 0.0
    env._state[4] = 0.0
    env.step(np.array([1.0], dtype=np.float32))
    assert env._v > 0.0


def test_negative_action_decreases_speed(env):
    env.reset()
    for _ in range(10):
        env.step(np.array([1.0], dtype=np.float32))
    v_before = env._v
    env.step(np.array([-1.0], dtype=np.float32))
    assert env._v < v_before


def test_speed_never_negative(env):
    env.reset()
    env._v = 0.0
    env._state[4] = 0.0
    for _ in range(5):
        env.step(np.array([-1.0], dtype=np.float32))
    assert env._v >= 0.0


# --- filter state ---

def test_filter_state_reset(env):
    env.reset()
    for _ in range(10):
        env.step(np.array([1.0], dtype=np.float32))
    env.reset()
    assert env._filtered_a == pytest.approx(0.0)
    assert env._filtered_jerk == pytest.approx(0.0)
    assert env._prev_action == pytest.approx(0.0)
    assert env._prev_a == pytest.approx(0.0)


# --- curvature setter ---

def test_set_curvature(env):
    env.reset()
    env.set_curvature(0.1)
    assert env._curvature == pytest.approx(0.1)


def test_set_curvature_affects_obs(env):
    env.reset()
    env.set_curvature(0.0)
    obs_flat, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    env.reset()
    env.set_curvature(0.4)
    obs_curved, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    # curvature obs is index 13 in default config
    assert obs_flat[13] != obs_curved[13]


# --- episode termination ---

def test_episode_terminates_after_max_steps(env):
    env.reset()
    terminated = False
    truncated = False
    for _ in range(env._max_episode_steps + 10):
        _, _, terminated, truncated, _ = env.step(np.array([0.5], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated or truncated

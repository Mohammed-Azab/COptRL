"""Tests for QuarterCarEnv gymnasium environment."""
import numpy as np
import pytest
import sys
import warnings
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register, registry

sys.path.insert(0, str(Path(__file__).parents[1] / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from quarter_car_core.quarter_car_env import QuarterCarEnv

ENV_ID = 'QuarterCarEnv-v0'
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point='quarter_car_core.quarter_car_env:QuarterCarEnv')


def make_env(profile='speed_bump', render_mode='none'):
    return QuarterCarEnv(road_profile=profile, render_mode=render_mode)


def test_gymnasium_env_checker():
    """gymnasium.utils.env_checker.check_env must pass."""
    from gymnasium.utils.env_checker import check_env
    env = gym.make(ENV_ID)
    check_env(env.unwrapped)
    env.close()


def test_reset_obs_within_bounds():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"Obs out of bounds after reset: {obs}"
    env.close()


def test_obs_shape():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,)
    env.close()


def test_action_space_shape():
    env = make_env()
    assert env.action_space.shape == (1,)
    env.close()


def test_step_zero_action_stays_in_bounds():
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.0], dtype=np.float32)
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"
    env.close()


def test_truncation_on_extreme_travel():
    env = make_env()
    env.reset(seed=0)
    # Force large suspension travel
    env._state = np.array([0.5, 0.0, 0.0, 0.0])
    _, _, _, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert truncated, "Should truncate when |z_s| > 0.3 m"
    env.close()


def test_info_dict_has_required_keys():
    env = make_env()
    _, info = env.reset(seed=0)
    required = {'rms_accel', 'peak_accel', 'suspension_rms',
                'comfort_score', 'road_profile', 'step_count', 'episode_time'}
    missing = required - set(info.keys())
    assert not missing, f"Missing info keys: {missing}"
    env.close()


def test_render_rgb_array():
    env = make_env(render_mode='rgb_array')
    env.reset(seed=0)
    env.step(np.array([0.0], dtype=np.float32))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unable to import Axes3D.*')
        frame = env.render()
    assert isinstance(frame, np.ndarray), "rgb_array render should return numpy array"
    assert frame.ndim == 3, "Frame should be H x W x C"
    env.close()


def test_episode_runs_to_completion():
    """Full 500-step episode with flat road and zero action should terminate (not truncate)."""
    env = QuarterCarEnv(road_profile='flat')
    env.reset(seed=0)
    action = np.array([0.0], dtype=np.float32)
    done = False
    steps = 0
    terminated = False
    truncated = False
    while not done:
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        if steps > 600:
            break
    assert terminated and not truncated, \
        f"Flat road, zero action should terminate cleanly (steps={steps})"
    env.close()


def test_get_comfort_metric():
    env = make_env()
    env.reset(seed=0)
    metric = env.get_comfort_metric()
    assert 0.0 <= metric <= 1.0
    env.close()

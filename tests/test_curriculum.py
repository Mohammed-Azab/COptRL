import numpy as np
import sys
sys.path.insert(0, "src/road")
import gymnasium as gym
import QuarterCar_env.envs  # noqa
from QuarterCar_env.wrappers import PreviewWrapper, CurriculumWrapper, load_curriculum_config

_CFG_PATH = "config/curriculum/curriculum_params.yaml"
ENV_ID    = "QuarterCar_env/QuarterCar"


def _make_curriculum(n_envs=1):
    cfg = load_curriculum_config(_CFG_PATH)
    env = gym.make(ENV_ID, road_profile="speed_bump")
    env = PreviewWrapper(env)
    env = CurriculumWrapper(env, cfg, n_envs=n_envs)
    return env, cfg


def test_starts_at_level_0():
    env, _ = _make_curriculum()
    assert env.current_level == 0
    env.close()


def test_level_advances_after_threshold():
    env, cfg = _make_curriculum(n_envs=1)
    threshold = cfg["thresholds"][0]
    # simulate enough steps to cross the first threshold
    env.reset(seed=0)
    for _ in range(threshold):
        env._step_count += 1   # advance directly to avoid running the full env
    assert env.current_level == 1
    env.close()


def test_reset_injects_level_road_kwargs():
    env, cfg = _make_curriculum()
    env.reset(seed=42)
    # level 0: max 2 bumps
    base = env.unwrapped.unwrapped   # CurriculumWrapper → PreviewWrapper → QuarterCarEnv
    assert len(base._road._bumps) <= cfg["levels"][0]["num_bumps_range"][1]
    env.close()


def test_different_levels_produce_different_ranges():
    cfg = load_curriculum_config(_CFG_PATH)

    # level 0 max height
    h_max_0 = cfg["levels"][0]["bump_height_range"][1]
    # level 2 max height
    h_max_2 = cfg["levels"][2]["bump_height_range"][1]
    assert h_max_2 > h_max_0, "harder level should allow taller bumps"


def test_obs_shape_unchanged_by_curriculum():
    env, _ = _make_curriculum()
    obs, _ = env.reset(seed=0)
    from QuarterCar_env.config.reward_params import load_reward_config
    rcfg = load_reward_config()
    assert obs.shape == (6 + 3 * rcfg.n_peaks,)
    env.close()

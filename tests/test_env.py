import numpy as np
import sys
sys.path.insert(0, "src/road")
import gymnasium as gym
import QuarterCar_env.envs  # noqa: registers the env
from QuarterCar_env.wrappers import PreviewWrapper

ENV_ID = "QuarterCar_env/QuarterCar"


def _make(road="speed_bump", **kwargs):
    env = gym.make(ENV_ID, road_profile=road, **kwargs)
    return PreviewWrapper(env)


def test_obs_shape_speed_bump():
    env = _make("speed_bump")
    obs, _ = env.reset(seed=0)
    from QuarterCar_env.config.reward_params import load_reward_config
    cfg = load_reward_config()
    expected = 6 + 3 * cfg.n_peaks
    assert obs.shape == (expected,), f"got {obs.shape}, expected ({expected},)"
    env.close()



def test_obs_finite_after_step():
    env = _make("speed_bump")
    env.reset(seed=1)
    obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert np.all(np.isfinite(obs)), "obs contains non-finite values"
    env.close()


def test_obs_in_bounds():
    env = _make("speed_bump")
    obs, _ = env.reset(seed=2)
    lo = env.observation_space.low
    hi = env.observation_space.high
    assert np.all(obs >= lo - 1e-5), f"obs below low: {obs[obs < lo - 1e-5]}"
    assert np.all(obs <= hi + 1e-5), f"obs above high: {obs[obs > hi + 1e-5]}"
    env.close()


def test_preview_slots_non_negative():
    env = _make("speed_bump")
    obs, _ = env.reset(seed=3)
    preview = obs[6:]   # base env gives 6 scalars; wrapper appends n_peaks*3
    assert np.all(preview >= 0.0), f"negative preview values: {preview}"
    env.close()


def test_random_reset_changes_bumps():
    env = _make("speed_bump", random_road_on_reset=True)
    env.reset(seed=0)
    bumps_0 = list(env.unwrapped._road._bumps)
    env.reset(seed=1)
    bumps_1 = list(env.unwrapped._road._bumps)
    assert bumps_0 != bumps_1, "two different seeds should produce different bump layouts"
    env.close()


def test_no_random_reset_keeps_fixed_layout():
    env = _make("speed_bump", random_road_on_reset=False)
    env.reset(seed=0)
    bumps_0 = list(env.unwrapped._road._bumps)
    env.reset(seed=99)
    bumps_1 = list(env.unwrapped._road._bumps)
    assert bumps_0 == bumps_1, "non-random mode should keep the same bump layout"
    env.close()


def test_options_can_disable_randomization():
    env = _make("speed_bump", random_road_on_reset=True)
    env.reset(seed=5, options={"randomize_road": False})
    bumps_0 = list(env.unwrapped._road._bumps)
    env.reset(seed=6, options={"randomize_road": False})
    bumps_1 = list(env.unwrapped._road._bumps)
    assert bumps_0 == bumps_1
    env.close()


def test_speed_is_clamped_to_geometry():
    env = _make("speed_bump", random_road_on_reset=True)
    env.reset(seed=7)
    v = env.unwrapped._v
    assert v > 0.0
    assert v <= env.unwrapped._rcfg.v_max
    env.close()


def test_flat_profile_unaffected_by_random_flag():
    env = _make("flat", random_road_on_reset=True)
    obs0, _ = env.reset(seed=0)
    obs1, _ = env.reset(seed=0)
    # flat road + same seed → identical obs
    assert np.allclose(obs0, obs1)
    env.close()

import gymnasium as gym
import yaml
from pathlib import Path


class CurriculumWrapper(gym.Wrapper):
    # adjusts road difficulty based on total env steps seen

    def __init__(self, env: gym.Env, config: dict, n_envs: int = 1):
        super().__init__(env)
        self._levels     = config["levels"]
        self._thresholds = config["thresholds"]
        self._n_envs     = n_envs   # each step() advances counter by n_envs
        self._step_count = 0

    def step(self, action):
        self._step_count += self._n_envs
        return super().step(action)

    def reset(self, **kwargs):
        level   = self._current_level()
        lvl_cfg = self._levels[level]

        opts = kwargs.get("options") or {}
        opts["randomize_road"]  = True
        opts["randomize_speed"] = True
        opts["road_kwargs"] = {
            "num_bumps_range":   tuple(lvl_cfg["num_bumps_range"]),
            "bump_height_range": tuple(lvl_cfg["bump_height_range"]),
            "bump_length_range": tuple(lvl_cfg["bump_length_range"]),
            "min_gap":           float(lvl_cfg["min_gap"]),
            "flat_start":        float(lvl_cfg["flat_start"]),
        }
        opts["v_random_low"]  = float(lvl_cfg["v_random_low"])
        opts["v_random_high"] = float(lvl_cfg["v_random_high"])
        kwargs["options"] = opts
        return super().reset(**kwargs)

    def _current_level(self) -> int:
        for i, threshold in enumerate(self._thresholds):
            if self._step_count < threshold:
                return i
        return len(self._thresholds)   # beyond last threshold → hardest level

    @property
    def current_level(self) -> int:
        return self._current_level()


def load_curriculum_config(path: str | Path) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    # yaml keys are strings; convert level keys to int
    raw["levels"] = {int(k): v for k, v in raw["levels"].items()}
    return raw

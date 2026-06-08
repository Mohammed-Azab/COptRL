import gymnasium as gym
import yaml
from pathlib import Path
from typing import Optional


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum wrapper — difficulty level is controlled externally by
    PerformanceCurriculumCallback, which only advances the level when the
    agent has demonstrated mastery (mean eval return ≥ threshold for N evals).

    Two modes of level control:
      set_level(n)        — training env: permanent one-way advance
      set_forced_level(n) — eval env: mirror training level for fair comparison
    """

    def __init__(self, env: gym.Env, config: dict, n_envs: int = 1):
        super().__init__(env)
        self._levels              = config["levels"]
        self._max_level           = len(self._levels) - 1
        self._active_level: int   = 0          # controlled by callback (training env)
        self._forced_level: Optional[int] = None  # controlled by sync callback (eval env)

    # ------------------------------------------------------------------
    # Level control API
    # ------------------------------------------------------------------

    def set_level(self, level: int) -> None:
        """Advance training level — one-way, called by PerformanceCurriculumCallback."""
        self._active_level = min(max(self._active_level, int(level)), self._max_level)

    def set_forced_level(self, level: Optional[int]) -> None:
        """Pin to a specific level — used by VecNormalizeSyncCallback on eval env."""
        self._forced_level = level if level is None else min(int(level), self._max_level)

    @property
    def current_level(self) -> int:
        """Return the level that the next reset() will use."""
        if self._forced_level is not None:
            return self._forced_level
        return self._active_level

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        lvl_cfg = self._levels[self.current_level]

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
        opts["v_random_low"]  = float(lvl_cfg["v_random_low"])  / 3.6   # km/h → m/s
        opts["v_random_high"] = float(lvl_cfg["v_random_high"]) / 3.6   # km/h → m/s
        kwargs["options"] = opts
        return super().reset(**kwargs)


def load_curriculum_config(path: str | Path) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw["levels"] = {int(k): v for k, v in raw["levels"].items()}
    # advance_return_threshold keys are also ints in the YAML
    if "advance_return_threshold" in raw:
        raw["advance_return_threshold"] = {
            int(k): float(v)
            for k, v in raw["advance_return_threshold"].items()
        }
    return raw

from __future__ import annotations

import numpy as np
import gymnasium as gym
import yaml
from pathlib import Path
from typing import Optional


class CurriculumWrapper(gym.Wrapper):
    # level controlled externally by PerformanceCurriculumCallback

    def __init__(self, env: gym.Env, config: dict, n_envs: int = 1):
        super().__init__(env)
        self._levels              = config["levels"]
        self._max_level           = len(self._levels) - 1
        self._active_level: int   = 0
        self._forced_level: Optional[int] = None
        self._rng                 = np.random.default_rng()
        self._scenarios: dict[int, list[dict]] = self._load_scenarios(config)

    def set_level(self, level: int) -> None:
        # one-way advance, training env, called by PerformanceCurriculumCallback
        self._active_level = min(max(self._active_level, int(level)), self._max_level)

    def set_forced_level(self, level: Optional[int]) -> None:
        # pin to level, eval env sync via VecNormalizeSyncCallback
        self._forced_level = level if level is None else min(int(level), self._max_level)

    @property
    def current_level(self) -> int:
        if self._forced_level is not None:
            return self._forced_level
        return self._active_level

    @staticmethod
    def _load_scenarios(config: dict) -> dict[int, list[dict]]:
        levels    = config["levels"]
        scenarios: dict[int, list[dict]] = {i: [] for i in range(len(levels))}

        sd_str = config.get("scenarios_dir")
        if not sd_str:
            return scenarios
        sd = Path(sd_str)
        if not sd.is_dir():
            return scenarios

        for lvl_idx, lvl_cfg in levels.items():
            diff = lvl_cfg.get("difficulty")
            if not diff:
                continue
            diff_dir = sd / diff
            if not diff_dir.is_dir():
                continue
            pool: list[dict] = []
            for f in sorted(diff_dir.glob("scenario_*.yaml")):
                with open(f) as fh:
                    pool.append(yaml.safe_load(fh))
            scenarios[lvl_idx] = pool

        return scenarios

    def reset(self, **kwargs):
        lvl     = self.current_level
        lvl_cfg = self._levels[lvl]
        pool    = self._scenarios.get(lvl, [])

        # prefer pre-generated single-bump scenario; fall back to random road
        use_scenario = bool(pool)
        if use_scenario and lvl_cfg.get("allow_multi_bump", False):
            if self._rng.random() < float(lvl_cfg.get("multi_bump_prob", 0.25)):
                use_scenario = False

        opts = kwargs.get("options") or {}

        if use_scenario:
            sc = pool[int(self._rng.integers(0, len(pool)))]
            v  = sc["speed_kmh"] / 3.6
            opts["randomize_road"]  = True
            opts["randomize_speed"] = True
            opts["road_kwargs"] = {
                "num_bumps_range": (1, 1),
                "catalog_ids":     [sc["catalog_id"]],
                "min_gap":         1.0,
                "max_gap":         1.0,
                "flat_start":      sc["flat_start_m"],
            }
            opts["v_random_low"]  = v
            opts["v_random_high"] = v
        else:
            opts["randomize_road"]  = True
            opts["randomize_speed"] = True
            opts["road_kwargs"] = {
                "num_bumps_range": tuple(lvl_cfg["num_bumps_range"]),
                "catalog_ids":     list(lvl_cfg["catalog_ids"]),
                "min_gap":         float(lvl_cfg["min_gap"]),
                "max_gap":         float(lvl_cfg["max_gap"]),
                "flat_start":      float(lvl_cfg["flat_start"]),
            }
            opts["v_random_low"]  = float(lvl_cfg["v_random_low"])  / 3.6
            opts["v_random_high"] = float(lvl_cfg["v_random_high"]) / 3.6

        kwargs["options"] = opts
        return super().reset(**kwargs)


def load_curriculum_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw["levels"] = {int(k): v for k, v in raw["levels"].items()}
    if "advance_return_threshold" in raw:
        raw["advance_return_threshold"] = {
            int(k): float(v)
            for k, v in raw["advance_return_threshold"].items()
        }
    # resolve scenarios_dir relative to repo root (curriculum_params.yaml is at
    # repo/config/curriculum/, so parents[2] = repo root)
    if "scenarios_dir" in raw:
        sd = raw["scenarios_dir"]
        if not Path(sd).is_absolute():
            raw["scenarios_dir"] = str(path.parents[2] / sd)
    # merge fallback random-road params from curr_multi_bumps.yaml
    multi_path = path.parent / "curr_multi_bumps.yaml"
    if multi_path.is_file():
        with open(multi_path) as fh:
            multi = yaml.safe_load(fh)
        multi_levels = {int(k): v for k, v in multi.get("levels", {}).items()}
        for lvl_idx, lvl_cfg in raw["levels"].items():
            if lvl_idx in multi_levels:
                for k, v in multi_levels[lvl_idx].items():
                    lvl_cfg.setdefault(k, v)
    return raw

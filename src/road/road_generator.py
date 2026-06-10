from __future__ import annotations
from typing import Optional
import json
from pathlib import Path
import numpy as np

from QuarterCar_env.config.road_params import MULTI_BUMP_CONFIG

def _get_catalog() -> list:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        p = parent / 'config' / 'road' / 'speed_bumps.json'
        if p.is_file():
            return json.loads(p.read_text())['bumps']
    raise FileNotFoundError('speed_bumps.json not found in config/road/')


class RoadGenerator:
    def __init__(self, profile: str = 'speed_bump', vehicle_speed: float = 10.0,
                 params: Optional[dict] = None):
        self.profile = profile
        self.speed   = vehicle_speed

        # Multi-bump layout for speed_bump profile.
        if params and all(k in params for k in ('bump_height', 'bump_length', 'bump_x_start')):
            self._bumps = [(
                float(params['bump_x_start']),
                float(params['bump_height']),
                float(params['bump_length']),
            )]
        else:
            mc = (params or {}).get('multi_bump_config', MULTI_BUMP_CONFIG)
            self._bumps = self._build_bumps(mc)

        # recorded profile state
        self._rec_arc:  Optional[np.ndarray] = None
        self._rec_z:    Optional[np.ndarray] = None
        self._rec_dzdx: Optional[np.ndarray] = None

    @staticmethod
    def _build_bumps(config: dict) -> list:
        catalog     = _get_catalog()
        bumps       = []
        x           = float(config["bump_x_start"])
        dis_mode    = config["dis_mode"]
        seq         = config["bump_sequence"]   # 0-based catalog IDs
        n           = min(int(config["num_bumps"]), len(seq))
        custom_gaps = config.get("custom_dis", [])
        constant_gap = float(config.get("constant_dis", 5.0))

        for i in range(n):
            ci = int(seq[i])
            A  = float(catalog[ci]['height_m'])
            L  = float(catalog[ci]['width_m'])
            bumps.append((x, A, L))
            if i < n - 1:
                if dis_mode == "custom" and i < len(custom_gaps):
                    gap = float(custom_gaps[i])
                else:
                    gap = constant_gap
                x += L + gap

        return bumps

    def get_height(self, t: float) -> float:
        # legacy, env uses get_height_at(s)
        return self.get_height_at(self.speed * t)

    def get_height_dot(self, t: float) -> float:
        # legacy, env uses get_height_dot_at(s, v)
        return self.get_height_dot_at(self.speed * t, self.speed)

    def get_height_at(self, s: float) -> float:
        # ζ(s): road height at arc-length position s [m]
        if self.profile == 'flat':
            return 0.0
        if self.profile == 'speed_bump':
            for x0, A, L in self._bumps:
                dx = s - x0
                if 0.0 <= dx <= L:
                    return (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L))
            return 0.0
        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_z is not None
            s_c = float(np.clip(s, self._rec_arc[0], self._rec_arc[-1]))
            return float(np.interp(s_c, self._rec_arc, self._rec_z))
        return 0.0

    def get_height_dot_at(self, s: float, v: float) -> float:
        # ζ̇(s, v) = dζ/dx · v: road velocity at arc-length s, vehicle speed v
        if self.profile == 'flat':
            return 0.0
        if self.profile == 'speed_bump':
            for x0, A, L in self._bumps:
                dx = s - x0
                if 0.0 < dx < L:
                    dzdx = (A / 2.0) * (2.0 * np.pi / L) * np.sin(2.0 * np.pi * dx / L)
                    return dzdx * v
            return 0.0
        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_dzdx is not None
            s_c  = float(np.clip(s, self._rec_arc[0], self._rec_arc[-1]))
            dzdx = float(np.interp(s_c, self._rec_arc, self._rec_dzdx))
            return dzdx * v
        return 0.0

    def get_height_array(self, t_array: np.ndarray) -> np.ndarray:
        # vectorised time-based query, legacy, kept for tests
        t = np.asarray(t_array, dtype=np.float64)
        return self.get_height_array_pos(self.speed * t)

    def get_height_array_pos(self, s_array: np.ndarray) -> np.ndarray:
        # vectorised position-based query used by the renderer
        s = np.asarray(s_array, dtype=np.float64)
        if self.profile == 'flat':
            return np.zeros(len(s))
        if self.profile == 'speed_bump':
            result = np.zeros(len(s))
            for x0, A, L in self._bumps:
                dx   = s - x0
                mask = (dx >= 0.0) & (dx <= L)
                result = np.where(
                    mask,
                    result + (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L)),
                    result,
                )
            return result
        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_z is not None
            s_c = np.clip(s, self._rec_arc[0], self._rec_arc[-1])
            return np.interp(s_c, self._rec_arc, self._rec_z)
        return np.zeros(len(s))

    def get_spatial_preview(
        self,
        s_pos: float,
        t_current: float,
        v_current: float,
        lookahead_m: float,
        n_points: int,
    ) -> np.ndarray:
        # Return road heights at n_points positions ahead of s_pos.
        s_offsets = np.linspace(lookahead_m / n_points, lookahead_m, n_points)

        if self.profile == 'flat':
            return np.zeros(n_points, dtype=np.float32)

        if self.profile == 'speed_bump':
            x_ahead = s_pos + s_offsets
            heights = np.zeros(n_points)
            for x0, A, L in self._bumps:
                dx   = x_ahead - x0
                mask = (dx >= 0.0) & (dx <= L)
                heights = np.where(
                    mask,
                    heights + (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L)),
                    heights,
                )
            return heights.astype(np.float32)

        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_z is not None
            x_ahead = np.clip(s_pos + s_offsets, self._rec_arc[0], self._rec_arc[-1])
            return np.interp(x_ahead, self._rec_arc, self._rec_z).astype(np.float32)

        return np.zeros(n_points, dtype=np.float32)

    def _clamp_speed_to_geometry(self) -> None:
        # limit v so peak ζ̇ = v·πH/L stays within ZETA_DOT_LIMIT (obs safe range)
        if self.profile != 'speed_bump' or not self._bumps:
            return
        ZETA_DOT_LIMIT = 7.0   # m/s, matches OBS_HIGH[1]
        steepest = max(np.pi * A / L for _, A, L in self._bumps)  # πH/L
        if steepest > 0:
            v_lim = ZETA_DOT_LIMIT / steepest
            self.speed = float(np.clip(self.speed, 0.0, v_lim))

    def set_speed(self, v: float) -> None:
        self.speed = float(v)
        self._clamp_speed_to_geometry()

    def reset(self, seed=None):
        pass   # nothing to regenerate for speed_bump / flat / recorded

    def load_recorded(self, arc_m: np.ndarray, z_m: np.ndarray) -> None:
        # Switch to 'recorded' profile using the supplied arc-length arrays.
        self._rec_arc  = np.asarray(arc_m, dtype=np.float64)
        self._rec_z    = np.asarray(z_m,   dtype=np.float64)
        self._rec_dzdx = np.gradient(self._rec_z, self._rec_arc)
        self.profile   = 'recorded'

    @classmethod
    def from_random(
        cls,
        rng: np.random.Generator,
        vehicle_speed: float,
        params: Optional[dict] = None,
        *,
        num_bumps_range: tuple = (1, 5),
        catalog_ids: Optional[list] = None,   # None = all catalog entries
        min_gap: float = 5.0,
        max_gap: float = 30.0,
        flat_start: float = 8.0,
    ) -> 'RoadGenerator':
        catalog  = _get_catalog()
        eligible = [catalog[i] for i in catalog_ids] if catalog_ids else catalog
        n    = int(rng.integers(num_bumps_range[0], num_bumps_range[1] + 1))
        idxs = rng.integers(0, len(eligible), size=n)
        gaps = rng.uniform(min_gap, max(max_gap, min_gap), size=max(n - 1, 0))

        # guarantee the first bump is always reachable without emergency braking
        _a_max = 5.0
        flat_start_safe = max(float(flat_start), vehicle_speed ** 2 / (2.0 * _a_max))
        bumps: list = []
        x = flat_start_safe
        for i, ci in enumerate(idxs):
            A = float(eligible[ci]['height_m'])
            L = float(eligible[ci]['width_m'])
            bumps.append((x, A, L))
            if i < n - 1:
                x += L + float(gaps[i])

        gen = cls(profile='speed_bump', vehicle_speed=vehicle_speed, params=params)
        gen._bumps = bumps
        gen._clamp_speed_to_geometry()
        return gen

    @classmethod
    def from_scenario_file(cls, path: str, speed: Optional[float] = None) -> 'RoadGenerator':
        # Load a scenario JSON file and return a ready RoadGenerator
        import json as _json
        with open(path) as fh:
            d = _json.load(fh)
        arc = np.array(d['arc_m'], dtype=np.float64)
        z   = np.array(d['z_m'],   dtype=np.float64)
        v   = float(d['v_ref']) if speed is None else float(speed)
        gen = cls(profile='recorded', vehicle_speed=v)
        gen.load_recorded(arc, z)
        return gen

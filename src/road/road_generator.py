from __future__ import annotations
from typing import Optional
import numpy as np

from QuarterCar_env.config.road_params import MULTI_BUMP_CONFIG


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
        """Return list of (x_start, height, length) tuples from multi-bump config."""
        bumps = []
        x = float(config["bump_x_start"])
        dis_mode    = config["dis_mode"]
        seq         = config["bump_sequence"]
        types       = config["bump_types"]
        n           = min(int(config["num_bumps"]), len(seq))
        custom_gaps = config.get("custom_dis", [])
        constant_gap = float(config.get("constant_dis", 5.0))

        for i in range(n):
            typ = int(seq[i])
            tp  = types[typ]
            A   = float(tp["bump_height"])
            L   = float(tp["bump_length"])
            bumps.append((x, A, L))
            if i < n - 1:
                if dis_mode == "custom" and i < len(custom_gaps):
                    gap = float(custom_gaps[i])
                else:
                    gap = constant_gap
                x += L + gap

        return bumps

    def get_height(self, t: float) -> float:
        if self.profile == 'flat':
            return 0.0
        if self.profile == 'speed_bump':
            x = self.speed * t
            for x0, A, L in self._bumps:
                dx = x - x0
                if 0.0 <= dx <= L:
                    return (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L))
            return 0.0
        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_z is not None
            x = np.clip(self.speed * t, self._rec_arc[0], self._rec_arc[-1])
            return float(np.interp(x, self._rec_arc, self._rec_z))
        return 0.0

    def get_height_dot(self, t: float) -> float:
        if self.profile == 'flat':
            return 0.0
        if self.profile == 'speed_bump':
            x = self.speed * t
            for x0, A, L in self._bumps:
                dx = x - x0
                if 0.0 < dx < L:
                    dzdx = (A / 2.0) * (2.0 * np.pi / L) * np.sin(2.0 * np.pi * dx / L)
                    return dzdx * self.speed
            return 0.0
        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_dzdx is not None
            x = np.clip(self.speed * t, self._rec_arc[0], self._rec_arc[-1])
            dzdx = float(np.interp(x, self._rec_arc, self._rec_dzdx))
            return dzdx * self.speed
        return 0.0

    def get_height_array(self, t_array: np.ndarray) -> np.ndarray:
        """Vectorised get_height for an array of query times (used by render)."""
        t = np.asarray(t_array, dtype=np.float64)
        if self.profile == 'flat':
            return np.zeros(len(t))
        if self.profile == 'speed_bump':
            result = np.zeros(len(t))
            x = self.speed * t
            for x0, A, L in self._bumps:
                dx   = x - x0
                mask = (dx >= 0.0) & (dx <= L)
                result = np.where(
                    mask,
                    result + (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L)),
                    result,
                )
            return result
        if self.profile == 'recorded':
            assert self._rec_arc is not None and self._rec_z is not None
            x = np.clip(self.speed * t, self._rec_arc[0], self._rec_arc[-1])
            return np.interp(x, self._rec_arc, self._rec_z)
        return np.zeros(len(t))

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
        """Clip speed so the peak road velocity ζ̇ stays within the observation window.

        For a cosine bump ζ(x) = H/2·(1−cos(2πx/L)), the peak spatial gradient is
        dζ/dx|_max = πH/L, giving ζ̇_max = v · πH/L.

        We limit ζ̇_max ≤ ZETA_DOT_LIMIT (the ODE/observation safe range) so that
        the most aggressive bump in the road stays within the model's valid regime.

        This replaces an earlier two-point empirical fit that erroneously clamped
        ALL bumps to ~20 km/h regardless of geometry (calibration points had nearly
        identical speeds, making the linear extrapolation meaningless).
        """
        if self.profile != 'speed_bump' or not self._bumps:
            return
        ZETA_DOT_LIMIT = 7.0   # m/s — matches OBS_HIGH[1] in env_params.yaml
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
        bump_height_range: tuple = (0.05, 0.25),
        bump_length_range: tuple = (1.0, 7.0),
        min_gap: float = 2.0,
        flat_start: float = 8.0,
    ) -> 'RoadGenerator':
        """Return a speed-bump RoadGenerator with randomly sampled geometry.

        All randomness uses the caller-supplied *rng* — no global np.random state is touched.
        Bumps are placed sequentially: first at *flat_start*, each subsequent one after the
        previous bump ends plus a uniformly sampled gap in [min_gap, 3·min_gap].
        Speed is automatically clamped to the geometry-safe limit after placement.
        """
        n       = int(rng.integers(num_bumps_range[0], num_bumps_range[1] + 1))
        heights = rng.uniform(*bump_height_range, size=n)
        lengths = rng.uniform(*bump_length_range, size=n)
        gaps    = rng.uniform(min_gap, min_gap * 3.0, size=max(n - 1, 0))

        bumps: list = []
        x = float(flat_start)
        for i in range(n):
            bumps.append((x, float(heights[i]), float(lengths[i])))
            if i < n - 1:
                x += float(lengths[i]) + float(gaps[i])

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

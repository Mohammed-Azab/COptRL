"""
Rule-based human driver baseline.

Models a cautious urban driver who:
  1. Scans ahead up to `preview_m` metres for speed bumps.
  2. For each visible bump, computes a comfortable crossing speed based on
     bump steepness (peak slope ζ̇_max = π·H/W).
  3. From current speed, computes the exact braking distance needed to arrive
     at that crossing speed at comfortable deceleration.
  4. Inside that zone, follows a kinematically correct linear speed ramp
     (constant deceleration), then re-accelerates after clearing the bump.

Parameters are loaded from config/baseline/human_driver_params.yaml so they
stay in sync with the rest of the project config.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml

if TYPE_CHECKING:
    from road.road_generator import RoadGenerator


def _load_params() -> dict:
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / 'config' / 'baseline' / 'human_driver_params.yaml'
        if p.is_file():
            with open(p) as fh:
                return yaml.safe_load(fh)
    return {}


class HumanDriverController:
    def __init__(
        self,
        v_max:          Optional[float] = None,
        v_min:          Optional[float] = None,
        a_max:          Optional[float] = None,
        preview_m:      Optional[float] = None,
        zeta_dot_limit: Optional[float] = None,
        a_brake:        Optional[float] = None,
        a_accel:        Optional[float] = None,
        ctrl_horizon:   Optional[float] = None,
    ):
        p = _load_params()

        # reward/physics limits come from the env config at call time when None
        self.v_max          = v_max
        self.v_min          = v_min
        self.a_max          = a_max
        self.preview_m      = preview_m      if preview_m      is not None else float(p.get('preview_m',      40.0))
        self.zeta_dot_limit = zeta_dot_limit if zeta_dot_limit is not None else float(p.get('zeta_dot_limit', 1.5))
        self.a_brake        = a_brake        if a_brake        is not None else float(p.get('a_brake',        2.5))
        self.a_accel        = a_accel        if a_accel        is not None else float(p.get('a_accel',        2.0))
        self.ctrl_horizon   = ctrl_horizon   if ctrl_horizon   is not None else float(p.get('ctrl_horizon',   1.2))

    # ------------------------------------------------------------------
    # Speed planning
    # ------------------------------------------------------------------

    def _crossing_speed(self, H: float, W: float) -> float:
        """Target speed so that peak road velocity ζ̇ = π·H/W·v ≤ zeta_dot_limit."""
        peak_slope = np.pi * H / W
        if peak_slope < 1e-6:
            return self.v_max
        return float(np.clip(self.zeta_dot_limit / peak_slope, self.v_min, self.v_max))

    def _target_speed(self, s_pos: float, v: float, road: 'RoadGenerator') -> float:
        """Return the most restrictive speed target given current position and speed."""
        bumps = getattr(road, '_bumps', [])
        if not bumps:
            return self.v_max

        v_target = self.v_max

        for x0, A, L in bumps:
            if s_pos >= x0 + L:
                # bump fully behind
                continue

            v_cross = self._crossing_speed(A, L)

            if s_pos >= x0:
                # on the bump — hold crossing speed
                v_target = min(v_target, v_cross)
                continue

            d_to_start = x0 - s_pos  # > 0

            if d_to_start > self.preview_m:
                continue

            if v <= v_cross:
                # already at or below crossing speed — nothing to do
                continue

            # braking distance from *current* speed to v_cross at comfortable decel
            d_brake = (v ** 2 - v_cross ** 2) / (2.0 * self.a_brake)

            if d_to_start >= d_brake:
                # not yet in braking zone
                continue

            # inside braking zone: kinematically correct linear speed ramp
            # v(d) = sqrt(v_cross² + 2·a_brake·d)
            v_t = float(np.sqrt(max(0.0, v_cross ** 2 + 2.0 * self.a_brake * d_to_start)))
            v_target = min(v_target, v_t)

        return max(self.v_min, v_target)

    # ------------------------------------------------------------------
    # Control interface — matches MPCController.act signature
    # ------------------------------------------------------------------

    def act(self, x: np.ndarray, s_pos: float, road: 'RoadGenerator') -> float:
        """Return normalised action u ∈ [-1, 1]."""
        v     = float(x[4])
        v_max = self.v_max if self.v_max is not None else v
        v_min = self.v_min if self.v_min is not None else 0.0
        a_max = self.a_max if self.a_max is not None else 5.0

        # temporarily bind for _crossing_speed
        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max

        # Direct braking decision: apply -a_brake when inside braking zone.
        # A P-controller on speed error gives near-zero action at braking onset
        # (v_t ≈ v at d == d_brake), so we detect the zone and command full decel.
        needs_brake = False
        bumps = getattr(road, '_bumps', [])
        for x0, A, L in bumps:
            if s_pos >= x0 + L:
                continue
            v_cross = self._crossing_speed(A, L)
            if s_pos >= x0:
                # on the bump — brake if still too fast
                if v > v_cross:
                    needs_brake = True
                continue
            d_to_start = x0 - s_pos
            if d_to_start > self.preview_m or v <= v_cross:
                continue
            d_brake = (v ** 2 - v_cross ** 2) / (2.0 * self.a_brake)
            if d_to_start <= d_brake:
                needs_brake = True

        if needs_brake:
            a_desired = -self.a_brake
        else:
            # gentle re-acceleration toward v_max
            a_desired = min(self.a_accel, (v_max - v) / self.ctrl_horizon)

        a_desired = float(np.clip(a_desired, -self.a_brake, self.a_accel))
        u = float(np.clip(a_desired / a_max, -1.0, 1.0))
        return u

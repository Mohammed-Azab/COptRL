"""
Rule-based human driver baseline.

Models a cautious urban driver who:
  1. Scans ahead up to `preview_m` metres for speed bumps.
  2. For each visible bump, computes a comfortable crossing speed based on
     bump steepness (peak slope ζ̇_max = π·H/W).
  3. Starts braking at the kinematically correct distance to arrive at that
     speed using a comfortable deceleration of `a_brake` m/s².
  4. Holds a linear speed ramp through the braking zone, crosses the bump,
     then re-accelerates back to cruise speed.

The resulting speed profile is identical to what a driver-model textbook would
call a "look-ahead proportional speed planner" — no optimisation, no model
integration, just geometry + kinematics.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from road.road_generator import RoadGenerator


class HumanDriverController:
    def __init__(
        self,
        v_max:       float = 20.0,   # m/s — cruise speed
        v_min:       float = 2.0,    # m/s — minimum crossing speed
        a_max:       float = 5.0,    # m/s² — max actuator authority
        a_brake:     float = 2.5,    # m/s² — comfortable deceleration
        a_accel:     float = 2.0,    # m/s² — comfortable re-acceleration
        preview_m:   float = 40.0,   # m  — how far ahead the driver looks
        # target peak road velocity ζ̇ = π·H/W·v_cross the driver tolerates [m/s]
        zeta_dot_limit: float = 1.5,
        # speed-control gain: time (s) over which the error is closed
        ctrl_horizon: float = 1.2,
    ):
        self.v_max          = v_max
        self.v_min          = v_min
        self.a_max          = a_max
        self.a_brake        = a_brake
        self.a_accel        = a_accel
        self.preview_m      = preview_m
        self.zeta_dot_limit = zeta_dot_limit
        self.ctrl_horizon   = ctrl_horizon

    # ------------------------------------------------------------------
    # Speed planning
    # ------------------------------------------------------------------

    def _crossing_speed(self, bump_height_m: float, bump_width_m: float) -> float:
        """Target speed to cross a bump with peak road velocity ≤ zeta_dot_limit."""
        peak_slope = np.pi * bump_height_m / bump_width_m   # ζ̇ = slope * v
        if peak_slope < 1e-6:
            return self.v_max
        v_cross = self.zeta_dot_limit / peak_slope
        return float(np.clip(v_cross, self.v_min, self.v_max))

    def _target_speed(self, s_pos: float, road: 'RoadGenerator') -> float:
        """Scan visible bumps and return the most restrictive speed target."""
        bumps = getattr(road, '_bumps', [])
        if not bumps:
            return self.v_max

        v_target = self.v_max

        for x0, A, L in bumps:
            bump_end = x0 + L

            if s_pos >= bump_end:
                # fully behind us
                continue

            v_cross = self._crossing_speed(A, L)

            if s_pos >= x0:
                # currently on the bump — hold crossing speed
                v_target = min(v_target, v_cross)
                continue

            d_to_start = x0 - s_pos   # > 0: approaching

            if d_to_start > self.preview_m:
                # outside scan window
                continue

            # braking distance needed to decelerate from v_max to v_cross
            # at comfortable deceleration: d = (v_max² - v_cross²) / (2·a_brake)
            d_brake = max(0.0, (self.v_max ** 2 - v_cross ** 2) / (2.0 * self.a_brake))

            if d_to_start >= d_brake:
                # not yet in braking zone — maintain cruise
                continue

            # inside braking zone: interpolate speed linearly from v_max → v_cross
            # progress=1.0 at braking-zone entry, 0.0 at bump face
            progress = d_to_start / max(d_brake, 1e-3)
            v_t = v_cross + progress * (self.v_max - v_cross)
            v_target = min(v_target, v_t)

        return max(self.v_min, v_target)

    # ------------------------------------------------------------------
    # Control interface (matches MPCController.act signature)
    # ------------------------------------------------------------------

    def act(self, x: np.ndarray, s_pos: float, road: 'RoadGenerator') -> float:
        """Return normalised action u ∈ [-1, 1]."""
        v = float(x[4])
        v_target = self._target_speed(s_pos, road)

        # proportional speed controller — close error over ctrl_horizon seconds
        a_desired = (v_target - v) / self.ctrl_horizon

        # clip to physical limits and normalise
        a_desired = float(np.clip(a_desired, -self.a_brake, self.a_accel))
        u = a_desired / self.a_max
        return float(np.clip(u, -1.0, 1.0))

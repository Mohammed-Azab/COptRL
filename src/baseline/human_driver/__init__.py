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
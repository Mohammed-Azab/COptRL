"""
QuarterCarODE — 6-state quarter-car physics engine.

  reset(v0)            → zero state at static equilibrium, speed = v0
  step(x, z_q_fn, t0) → RK4 over N_SUB substeps
                         returns (new_state, z_B_ddot, z_W_ddot)
"""

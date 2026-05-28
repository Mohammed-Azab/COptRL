"""
QuarterCarODE -> internal physics engine for the speed-control environment.

The ODE is a 6-state vehicle dynamics model driven by road input.
It is a black-box from the agent's perspective: the agent never sees
its internal state directly; only the road contact signals (ζ, ζ̇) and
the body vertical acceleration (used for the comfort reward) are exposed.

    reset(v0)  →  zero internal state at static equilibrium, speed = v0
    step(x, z_q_fn, t0)  →  RK4 over N_SUB substeps, returns (new_state, body_accel)
"""

"""
    func reset():
        Zero deflections at static equilibrium; longitudinal velocity = v0.

    func step():
        -> Integrate one control step (DT = N_SUB × DT_SIM) with RK4.
        -> Returns (new_state, z_B_ddot) 
        -> z_B_ddot is body acceleration at end-of-step (used for reward computation).
"""

# Quarter-car 2-DOF ODE model.

import numpy as np
from scipy.integrate import solve_ivp

from quarter_car_core.params import PHYSICS


class QuarterCarODE:
    def __init__(self, params: dict = None):
        p = {**PHYSICS, **(params or {})}
        self.m_s = p['m_s']
        self.m_u = p['m_u']
        self.k_s = p['k_s']
        self.c_s = p['c_s']
        self.k_t = p['k_t']
        self._internal_dt = 0.002  # 500 Hz

    def _ode(self, t, y, F_act, z_r):
        z_s, z_s_dot, z_u, z_u_dot = y
        susp = self.k_s * (z_s - z_u) + self.c_s * (z_s_dot - z_u_dot)
        z_s_ddot = (-susp + F_act) / self.m_s
        z_u_ddot = (susp - self.k_t * (z_u - z_r) - F_act) / self.m_u
        return [z_s_dot, z_s_ddot, z_u_dot, z_u_ddot]

    def step(self, state, F_act: float, z_r: float, dt: float = 0.02):
        """Integrate one control step at 500 Hz internally. Returns (new_state, z_s_ddot)."""
        sol = solve_ivp(
            self._ode, [0.0, dt], list(state),
            args=(float(F_act), float(z_r)),
            method='RK45', max_step=self._internal_dt,
        )
        new_state = sol.y[:, -1]
        derivs = self._ode(dt, new_state, F_act, z_r)
        return new_state, float(derivs[1])

    def get_state_space(self):
        """Linearised (A, B, C, D) about equilibrium for LQR/MPC design [1][5]."""
        ms, mu, ks, cs, kt = self.m_s, self.m_u, self.k_s, self.c_s, self.k_t
        A = np.array([
            [0,       1,          0,       0      ],
            [-ks/ms, -cs/ms,      ks/ms,   cs/ms  ],
            [0,       0,          0,       1      ],
            [ks/mu,   cs/mu,  -(ks+kt)/mu, -cs/mu ],
        ])
        B = np.array([[0.0], [1.0/ms], [0.0], [-1.0/mu]])
        C = np.eye(4)
        D = np.zeros((4, 1))
        return A, B, C, D

    def reset(self):
        return np.zeros(4, dtype=np.float64)

# Mandl (2021) quarter-car ODE: 6-state, nonlinear spring + PWA damper + RK4.
# Sections 3.1, 3.3, 3.4, 3.5.1  |  Appendix A.1

import numpy as np
from typing import Callable

from QuarterCar_env.params import PHYSICS, DT_SIM, N_SUB, VEHICLE_SPEED


class _P:
    """Flat parameter struct for the ODE hot-path (avoids dict look-ups)."""
    __slots__ = (
        'm_B', 'm_W', 'c_T', 'k_T', 'k_S', 'dz_S_stat',
        'd1', 'z1', 'd2', 'z2', 'v_d', 'v_z',
        'f1_cmp', 'f2_cmp', 'f1_rbd', 'f2_rbd',
        'dz_cmp', 'dz_rbd', 'F_ks_nlin_max',
    )

    def __init__(self, d: dict):
        for k in self.__slots__:
            setattr(self, k, float(d[k]))


def _spring_nonlin(dyn: float, p: _P) -> float:
    """
    Nonlinear bumpstop spring force (Eqs. 3.9–3.11).
    dyn = x[2] = z_W − z_B  (positive = compression, negative = rebound)
    Exponent form: exp(Δz_F · f2 / Δz_S_stat)  — note f2 multiplies, not divides.
    """
    if dyn > p.dz_cmp:               # compression bumpstop  (case 2 in Fig. 3.4b)
        dz_F     = dyn - p.dz_cmp
        exp_arg  = dz_F * p.f2_cmp / p.dz_S_stat
        F = p.k_S * (p.dz_S_stat * p.f1_cmp * (np.exp(exp_arg) - 1.0) - dz_F)
    elif dyn < -p.dz_rbd:            # rebound bumpstop      (case 3 in Fig. 3.4b)
        dz_F     = -dyn - p.dz_rbd
        exp_arg  = dz_F * p.f2_rbd / p.dz_S_stat
        F = -p.k_S * (p.dz_S_stat * p.f1_rbd * (np.exp(exp_arg) - 1.0) - dz_F)
    else:
        return 0.0                   # linear range           (case 1 in Fig. 3.4b)
    return float(np.clip(F, -p.F_ks_nlin_max, p.F_ks_nlin_max))


def _damper(v_S: float, p: _P) -> float:
    """
    Piecewise-affine asymmetric damper (Eq. 3.12).
    v_S = x[1] − x[3] = ż_W − ż_B  (positive = compression).
    Slopes: d1/d2 = compression (low/high speed), z1/z2 = rebound.
    """
    if v_S >= p.v_d:                 # high-speed compression  (case 3)
        return p.d2 * (v_S - p.v_d) + p.d1 * p.v_d
    elif v_S >= 0.0:                 # low-speed compression   (case 1)
        return p.d1 * v_S
    elif v_S >= -p.v_z:             # low-speed rebound       (case 2)
        return p.z1 * v_S
    else:                            # high-speed rebound      (case 4)
        return p.z2 * (v_S + p.v_z) - p.z1 * p.v_z


def _ode(x: np.ndarray, z_q: float, F_act: float, p: _P) -> np.ndarray:
    """
    6-state Mandl quarter-car equations of motion (Eqs. 3.21a–c, 3.22).
    Active suspension force F_act added on top of passive elements.

    State:  x = [ζ−z_W, ż_W, z_W−z_B, ż_B, v, z_B]
                 [  0     1     2        3   4   5  ]
    Input:  z_q = ζ̇  (road velocity disturbance)
            F_act    (active suspension force, + lifts body / − presses wheel)
    """
    F_spring = p.k_S * x[2] + _spring_nonlin(x[2], p)   # F_ks,lin + F_ks,nonlin
    F_damp   = _damper(x[1] - x[3], p)                   # F_cs  (v_S = ż_W − ż_B)
    F_tire_k = p.k_T * x[0]                               # k_T·(ζ − z_W)  (3.15d)
    F_tire_c = p.c_T * (z_q - x[1])                      # c_T·(ζ̇ − ż_W) (3.15e)

    dx = np.empty(6, dtype=np.float64)
    dx[0] = z_q - x[1]                                                    # (3.22)
    dx[1] = (-F_spring - F_damp - F_act + F_tire_k + F_tire_c) / p.m_W   # (3.21b)
    dx[2] = x[1] - x[3]                                                   # (3.22)
    dx[3] = ( F_spring + F_damp + F_act) / p.m_B                          # (3.21a)
    dx[4] = 0.0                                                            # const speed
    dx[5] = x[3]                                                           # ż_B integral
    return dx


class MandlQuarterCarODE:
    """Fixed-step RK4 integrator for the Mandl 6-state quarter-car model."""

    def __init__(self, params: dict = None):
        d = {**PHYSICS, **(params or {})}
        self._p = _P(d)

    def step(
        self,
        x: np.ndarray,
        z_q_fn: Callable[[float], float],
        t0: float,
        F_act: float,
    ) -> tuple[np.ndarray, float]:
        """
        Integrate one control step (DT = N_SUB × DT_SIM) with RK4.

        z_q_fn(t) → ζ̇ at time t  (road velocity, called at sub-step instants).
        Returns (new_state, z_B_ddot) where z_B_ddot is body vertical acceleration
        evaluated at the end of the step (used for reward and info).
        """
        dt = DT_SIM
        p  = self._p
        xi = x.copy()

        for i in range(N_SUB):
            t   = t0 + i * dt
            th  = t  + 0.5 * dt
            te  = t  + dt
            zq0 = float(z_q_fn(t))
            zqh = float(z_q_fn(th))
            zqe = float(z_q_fn(te))

            k1 = _ode(xi,               zq0, F_act, p)
            k2 = _ode(xi + 0.5*dt*k1,  zqh, F_act, p)
            k3 = _ode(xi + 0.5*dt*k2,  zqh, F_act, p)
            k4 = _ode(xi +     dt*k3,  zqe, F_act, p)
            xi = xi + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        zq_end   = float(z_q_fn(t0 + N_SUB * dt))
        derivs   = _ode(xi, zq_end, F_act, p)
        z_B_ddot = float(derivs[3])

        return xi, z_B_ddot

    def reset(self, v0: float = VEHICLE_SPEED) -> np.ndarray:
        """Zero deflections at static equilibrium; longitudinal velocity = v0."""
        x = np.zeros(6, dtype=np.float64)
        x[4] = v0
        return x

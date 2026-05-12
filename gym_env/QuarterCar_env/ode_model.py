# Quarter-car ODE: 6-state, nonlinear bumpstop spring + PWA damper, RK4 integrator.

import numpy as np
from typing import Callable

from QuarterCar_env.params import PHYSICS, DT_SIM, N_SUB, VEHICLE_SPEED


class _P:
    """Flat parameter struct to avoids dict look-ups on the ODE hot-path."""
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
    Exponential bumpstop force beyond the linear clearance zone.
    dyn = x[2] = z_W − z_B  (positive = compression, negative = rebound)
    """
    if dyn > p.dz_cmp:
        dz_F    = dyn - p.dz_cmp
        exp_arg = dz_F * p.f2_cmp / p.dz_S_stat
        F = p.k_S * (p.dz_S_stat * p.f1_cmp * (np.exp(exp_arg) - 1.0) - dz_F)
    elif dyn < -p.dz_rbd:
        dz_F    = -dyn - p.dz_rbd
        exp_arg = dz_F * p.f2_rbd / p.dz_S_stat
        F = -p.k_S * (p.dz_S_stat * p.f1_rbd * (np.exp(exp_arg) - 1.0) - dz_F)
    else:
        return 0.0
    return float(np.clip(F, -p.F_ks_nlin_max, p.F_ks_nlin_max))


def _damper(v_S: float, p: _P) -> float:
    """
    Piecewise-affine asymmetric damper.
    v_S = ż_W − ż_B  (positive = compression)
    Four regimes: low/high-speed × compression/rebound.
    """
    if v_S >= p.v_d:
        return p.d2 * (v_S - p.v_d) + p.d1 * p.v_d
    elif v_S >= 0.0:
        return p.d1 * v_S
    elif v_S >= -p.v_z:
        return p.z1 * v_S
    else:
        return p.z2 * (v_S + p.v_z) - p.z1 * p.v_z


def _ode(x: np.ndarray, z_q: float, F_act: float, p: _P) -> np.ndarray:
    """
    6-state quarter-car equations of motion.

    State:  x = [ζ−z_W, ż_W, z_W−z_B, ż_B, v, z_B]
    Input:  z_q  = ζ̇ (road velocity)
            F_act = active suspension force (+ lifts body / presses wheel down)
    """
    F_spring = p.k_S * x[2] + _spring_nonlin(x[2], p)
    F_damp   = _damper(x[1] - x[3], p)
    F_tire_k = p.k_T * x[0]
    F_tire_c = p.c_T * (z_q - x[1])

    dx = np.empty(6, dtype=np.float64)
    dx[0] = z_q - x[1]
    dx[1] = (-F_spring - F_damp - F_act + F_tire_k + F_tire_c) / p.m_W
    dx[2] = x[1] - x[3]
    dx[3] = ( F_spring + F_damp + F_act) / p.m_B
    dx[4] = 0.0   # longitudinal speed is constant
    dx[5] = x[3]
    return dx


class QuarterCarODE:
    """Fixed-step RK4 integrator for the 6-state quarter-car model."""

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
        Returns (new_state, z_B_ddot) where z_B_ddot is body acceleration
        at the end of the step (used for reward).
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
        x    = np.zeros(6, dtype=np.float64)
        x[4] = v0
        return x

"""
Quarter-car ODE: 6-state.

State vector layout
  x[0] = ζ − z_W   tyre deflection      (+ = compression)
  x[1] = ż_W       wheel velocity
  x[2] = z_W − z_B suspension travel    (+ = compression)
  x[3] = ż_B       body velocity
  x[4] = v         longitudinal speed   (driven by env, not ODE)
  x[5] = z_B       body displacement

"""
import numpy as np
from numba import njit
from typing import Callable

from QuarterCar_env.config.env_params import PHYSICS, DT_SIM, N_SUB
from QuarterCar_env.config.road_params import VEHICLE_SPEED


class _P:
    """Flat parameter struct for the quarter-car ODE model."""
    __slots__ = (
        # masses
        'm_B', 'm_W',
        # spring/damper parameters
        'c_T', 'k_T', 'k_S',
        # nonlinear spring parameters
        'dz_S_stat',
        'd1', 'z1', 'd2', 'z2', 'v_d', 'v_z',
        # nonlinear spring force limits
        'f1_cmp', 'f2_cmp', 'f1_rbd', 'f2_rbd',
        # nonlinear spring clearance limits
        'dz_cmp', 'dz_rbd',
        # nonlinear spring force limits
        'F_ks_nlin_max',
    )

    def __init__(self, d: dict):
        for k in self.__slots__:
            setattr(self, k, float(d[k]))


# parameter vector layout  
_I_MB,  _I_MW  = 0, 1
_I_CT,  _I_KT, _I_KS, _I_DZS = 2, 3, 4, 5
_I_D1,  _I_Z1, _I_D2, _I_Z2  = 6, 7, 8, 9
_I_VD,  _I_VZ                 = 10, 11
_I_F1C, _I_F2C, _I_F1R, _I_F2R = 12, 13, 14, 15
_I_DZC, _I_DZR, _I_FMAX       = 16, 17, 18
_P_LEN = 19

# state vector slot indices
_X_TYRE = 0   # ζ − z_W   tyre deflection      (+ = compression)
_X_ZW   = 1   # ż_W       wheel velocity
_X_SUSP = 2   # z_W − z_B suspension travel     (+ = compression)
_X_ZB   = 3   # ż_B       body velocity
_X_V    = 4   # v         longitudinal speed     (set by env, not ODE)
_X_POS  = 5   # z_B       body displacement


def _build_pvec(d: dict) -> np.ndarray:
    """Pack physics dict into a contiguous float64 array for Numba."""
    v = np.empty(_P_LEN, dtype=np.float64)
    v[_I_MB]  = d['m_B'];      v[_I_MW]  = d['m_W']
    v[_I_CT]  = d['c_T'];      v[_I_KT]  = d['k_T']
    v[_I_KS]  = d['k_S'];      v[_I_DZS] = d['dz_S_stat']
    v[_I_D1]  = d['d1'];       v[_I_Z1]  = d['z1']
    v[_I_D2]  = d['d2'];       v[_I_Z2]  = d['z2']
    v[_I_VD]  = d['v_d'];      v[_I_VZ]  = d['v_z']
    v[_I_F1C] = d['f1_cmp'];   v[_I_F2C] = d['f2_cmp']
    v[_I_F1R] = d['f1_rbd'];   v[_I_F2R] = d['f2_rbd']
    v[_I_DZC] = d['dz_cmp'];   v[_I_DZR] = d['dz_rbd']
    v[_I_FMAX]= d['F_ks_nlin_max']
    return v


# numba-jitted ODE kernels

@njit(cache=True)
def _spring_nonlin(dyn: float, p: np.ndarray) -> float:
    """
    Exponential bumpstop beyond the linear clearance zone.
    dyn = x[_X_SUSP] = z_W − z_B  (+ = compression, − = rebound)
    Returns 0 inside the linear zone; ramps exponentially outside.
    """
    dz_cmp = p[_I_DZC];   dz_rbd = p[_I_DZR]
    k_S    = p[_I_KS];    dz_s   = p[_I_DZS]
    F_max  = p[_I_FMAX]

    if dyn > dz_cmp:
        dz_F    = dyn - dz_cmp
        exp_arg = dz_F * p[_I_F2C] / dz_s
        F = k_S * (dz_s * p[_I_F1C] * (np.exp(exp_arg) - 1.0) - dz_F)
    elif dyn < -dz_rbd:
        dz_F    = -dyn - dz_rbd
        exp_arg = dz_F * p[_I_F2R] / dz_s
        F = -k_S * (dz_s * p[_I_F1R] * (np.exp(exp_arg) - 1.0) - dz_F)
    else:
        return 0.0

    # hard cap so numerical blow-up can't destabilise the integrator
    if F >  F_max: return  F_max
    if F < -F_max: return -F_max
    return F


@njit(cache=True)
def _damper(v_S: float, p: np.ndarray) -> float:
    """
    Smooth asymmetric damper via sigmoid-blended piecewise slopes.

    Compression regimes (v_S > 0): low-speed slope d1, high-speed slope d2.
    Rebound    regimes (v_S < 0): low-speed slope z1, high-speed slope z2.
    """
    d1 = p[_I_D1];  z1 = p[_I_Z1]
    d2 = p[_I_D2];  z2 = p[_I_Z2]
    v_d = p[_I_VD]; v_z = p[_I_VZ]
    k   = 50.0

    # blend d1 / z1 slopes across v_S = 0
    w_cmp  = 1.0 / (1.0 + np.exp(-k * v_S))
    F_base = (w_cmp * d1 + (1.0 - w_cmp) * z1) * v_S

    # extra compression slope (d2 - d1) kicks in past v_d
    w_hc = 1.0 / (1.0 + np.exp(-k * (v_S - v_d)))
    F_hc = w_hc * (d2 - d1) * (v_S - v_d)

    # extra rebound slope (z2 - z1) kicks in past -v_z
    w_hr = 1.0 / (1.0 + np.exp(-k * (-v_S - v_z)))
    F_hr = w_hr * (z2 - z1) * (-v_S - v_z)

    return F_base + F_hc - F_hr


@njit(cache=True)
def _ode(x: np.ndarray, z_q: float, p: np.ndarray) -> np.ndarray:
    """
    6-state quarter-car equations of motion.

    x = [ζ−z_W, ż_W, z_W−z_B, ż_B, v, z_B]
    z_q = ζ̇  (road velocity, m/s)

    """
    # suspension forces (body–wheel interface)
    F_spring = p[_I_KS] * x[_X_SUSP] + _spring_nonlin(x[_X_SUSP], p)
    F_damp   = _damper(x[_X_ZW] - x[_X_ZB], p)   # v_S = ż_W − ż_B

    # tyre forces (wheel–road interface)
    F_tire_k = p[_I_KT] * x[_X_TYRE]              # spring: k_T * (ζ − z_W)
    F_tire_c = p[_I_CT] * (z_q - x[_X_ZW])        # damper: c_T * (ζ̇ − ż_W)

    dx            = np.empty(6)
    dx[_X_TYRE]   = z_q - x[_X_ZW]                           # d/dt(ζ − z_W) = ζ̇ − ż_W
    dx[_X_ZW]     = (-F_spring - F_damp + F_tire_k + F_tire_c) / p[_I_MW]
    dx[_X_SUSP]   = x[_X_ZW] - x[_X_ZB]                     # d/dt(z_W − z_B) = ż_W − ż_B
    dx[_X_ZB]     = ( F_spring + F_damp) / p[_I_MB]
    dx[_X_V]      = 0.0                                        # speed driven by env, not ODE
    dx[_X_POS]    = x[_X_ZB]                                  # ż_B
    return dx


@njit(cache=True)
def _rk4_loop(x: np.ndarray, zq_pre: np.ndarray, dt: float, p: np.ndarray) -> np.ndarray:
    """
    Fixed-step RK4 over N_SUB substeps using pre-sampled road velocities.

    zq_pre: shape (N_SUB, 3) — columns are [z_q(t), z_q(t+dt/2), z_q(t+dt)]
    Road velocities are precomputed in Python so this loop is pure-numba.
    """
    xi = x.copy()
    n  = zq_pre.shape[0]
    for i in range(n):
        zq0 = zq_pre[i, 0]
        zqh = zq_pre[i, 1]
        zqe = zq_pre[i, 2]
        k1 = _ode(xi,               zq0, p)
        k2 = _ode(xi + 0.5*dt*k1,  zqh, p)
        k3 = _ode(xi + 0.5*dt*k2,  zqh, p)
        k4 = _ode(xi +     dt*k3,  zqe, p)
        xi = xi + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return xi


class QuarterCarODE:
    # ODE for the 6-state quarter-car model

    def __init__(self, params: dict = None):
        d = {**PHYSICS, **(params or {})}
        self._p    = _P(d)            # Python struct retained for inspection / testing
        self._pvec = _build_pvec(d)   # flat array passed to @njit kernels

    def step(
        self,
        x: np.ndarray,
        z_q_fn: Callable[[float], float],
        t0: float,
    ) -> tuple[np.ndarray, float]:
    
        dt = DT_SIM
        p  = self._pvec

        # sample road velocity at all RK4 evaluation points before entering numba
        zq_pre = np.empty((N_SUB, 3), dtype=np.float64)
        for i in range(N_SUB):
            t = t0 + i * dt
            zq_pre[i, 0] = float(z_q_fn(t))
            zq_pre[i, 1] = float(z_q_fn(t + 0.5 * dt))
            zq_pre[i, 2] = float(z_q_fn(t + dt))

        xi = _rk4_loop(x, zq_pre, dt, p)

        # body acceleration at the terminal state for the reward signal
        zq_end   = float(z_q_fn(t0 + N_SUB * dt))
        z_B_ddot = float(_ode(xi, zq_end, p)[3])

        # Returns (new_state, z_B_ddot) 
        return xi, z_B_ddot

    def reset(self, v0: float = VEHICLE_SPEED) -> np.ndarray:

        x    = np.zeros(6, dtype=np.float64)
        # Zero deflections at static equilibrium.
        x[4] = v0
        return x

import numpy as np

# ── Mandl (2021) Table A.1 — Quarter-Car Parameters ──────────────────────────
# Masses
m_B = 465.7   # kg — sprung mass (car body)
m_W = 50.4    # kg — unsprung mass (car wheel)

# Tire (linear spring–damper, Eqs. 3.15d–e)
c_T = 500.0       # N·s/m — tire damping coefficient
k_T = 262_200.0   # N/m   — tire stiffness

# Suspension spring — linear stiffness (Eq. 3.15b)
k_S = 27_922.0    # N/m   — 2.7922 × 10⁴

# Static equilibrium suspension deflection (Eq. 3.19): Δz_S,stat = m_B·g / k_S
dz_S_stat = m_B * 9.81 / k_S   # ≈ 0.1636 m — used in nonlinear spring exponent

# Suspension damper — slopes derived from Table A.1 (Sec. 3.4, page 104)
#   D = (d1+z1)/2 = 3530,  A = d1/z1 = 0.5
#   → z1 = 2D/(1+A),  d1 = A·z1
_D, _A = 3530.0, 0.5
z1 = 2.0 * _D / (1.0 + _A)     # 4706.67 N·s/m — low-speed rebound slope
d1 = _A * z1                    # 2353.33 N·s/m — low-speed compression slope
#   degression factors: s_d = d1/d2 = 0.25,  s_z = z1/z2 = 0.40
d2 = d1 / 0.25                  # 9413.33 N·s/m — high-speed compression slope
z2 = z1 / 0.40                  # 11766.67 N·s/m — high-speed rebound slope
# Transition velocities (low → high speed)
v_d = 0.20   # m/s — compression transition
v_z = 0.20   # m/s — rebound transition

# Nonlinear bumpstop spring (Table A.1, Eqs. 3.9–3.11)
f1_cmp = 1.0 / 3.0   # progression factor — compression
f2_cmp = 4.0          # exponent curvature — compression
f1_rbd = 1.0          # progression factor — rebound
f2_rbd = 8.0          # exponent curvature — rebound
dz_cmp = 0.02         # m — clearance before compression onset
dz_rbd = 0.08         # m — clearance before rebound onset
F_ks_nlin_max = 1e5   # N — hard clip on nonlinear spring force

# Physics parameter dict consumed by MandlQuarterCarODE
PHYSICS = {
    'm_B': m_B, 'm_W': m_W,
    'c_T': c_T, 'k_T': k_T, 'k_S': k_S,
    'dz_S_stat': dz_S_stat,
    'd1': d1, 'z1': z1, 'd2': d2, 'z2': z2,
    'v_d': v_d, 'v_z': v_z,
    'f1_cmp': f1_cmp, 'f2_cmp': f2_cmp,
    'f1_rbd': f1_rbd, 'f2_rbd': f2_rbd,
    'dz_cmp': dz_cmp, 'dz_rbd': dz_rbd,
    'F_ks_nlin_max': F_ks_nlin_max,
}

# ── Actuator ──────────────────────────────────────────────────────────────────
F_MAX = 10_000.0   # N — active suspension force limit

# ── Control loop timing ───────────────────────────────────────────────────────
DT            = 0.02          # s — control step (50 Hz)
DT_SIM        = 1e-3          # s — RK4 sub-step (1 kHz)
N_SUB         = int(DT / DT_SIM)   # 20 sub-steps per control step
EPISODE_STEPS = 500           # steps → 10 s per episode at 50 Hz

# ── Truncation thresholds ─────────────────────────────────────────────────────
TRUNC_TRAVEL = 0.10   # m — |z_W − z_B|  suspension travel
TRUNC_ZS     = 0.30   # m — |z_B|         body displacement from static eq.

# ── Observation space bounds (8-dim) ─────────────────────────────────────────
# idx: [z_B, ż_B, z_W, ż_W, ζ, ζ̇, travel, tyre_defl]
OBS_HIGH = np.array([0.30, 3.0, 0.40, 5.0, 0.15, 7.0, 0.15, 0.05], dtype=np.float32)
OBS_LOW  = -OBS_HIGH

# ── Road profile defaults ─────────────────────────────────────────────────────
ROAD_DEFAULTS = {
    'bump_height':      0.1,     # m
    'bump_length':      0.5,     # m
    'iso_gd0':          256e-6,  # m³/cycle — ISO 8608 Class C
    'iso_n0':           0.1,     # cycle/m
    'sweep_amplitude':  0.02,    # m
    'episode_duration': 10.0,    # s
}

VEHICLE_SPEED = 10.0   # m/s (36 km/h)

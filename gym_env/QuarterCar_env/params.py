import numpy as np

#  Quarter-Car Physical Parameters 

# Masses
m_B = 465.7   # kg - sprung mass (car body)
m_W = 50.4    # kg - unsprung mass (wheel assembly)

# Tire parameters (spring + damper)
c_T = 500.0       # N·s/m - tire damping
k_T = 262_200.0   # N/m   - tire stiffness

# Suspension spring
k_S = 27_922.0    # N/m

# Static equilibrium deflection: Δz_S,stat = m_B·g / k_S
dz_S_stat = m_B * 9.81 / k_S   

# Suspension damper (piecewise-affine, asymmetric compression vs rebound)
#   D = (d1+z1)/2 = 3530  (mean slope),  A = d1/z1 = 0.5  (asymmetry ratio)
#   → z1 = 2D/(1+A),  d1 = A·z1
_D, _A = 3530.0, 0.5
z1 = 2.0 * _D / (1.0 + _A)     # 4706.67 N·s/m - low-speed rebound slope
d1 = _A * z1                    # 2353.33 N·s/m - low-speed compression slope
#   degression factors: s_d = d1/d2 = 0.25,  s_z = z1/z2 = 0.40
d2 = d1 / 0.25                  # 9413.33 N·s/m - high-speed compression slope
z2 = z1 / 0.40                  # 11766.67 N·s/m - high-speed rebound slope
# velocity thresholds between low- and high-speed regimes
v_d = 0.20   # m/s - compression
v_z = 0.20   # m/s - rebound

# Nonlinear bumpstop spring (exponential progression beyond clearance)
f1_cmp = 1.0 / 3.0   # progression factor - compression
f2_cmp = 4.0          # exponent curvature - compression
f1_rbd = 1.0          # progression factor - rebound
f2_rbd = 8.0          # exponent curvature - rebound
dz_cmp = 0.02         # m - compression clearance
dz_rbd = 0.08         # m - rebound clearance
F_ks_nlin_max = 1e5   # N - hard clip

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

#  Actuator 
F_MAX = 10_000.0   # N - active suspension force limit

# Control loop timing
DT            = 0.02          # s 
DT_SIM        = 1e-3          # s - RK4 sub-step (1 kHz)
N_SUB         = int(DT / DT_SIM)
EPISODE_STEPS = 260           # steps 

# Truncation thresholds
TRUNC_TRAVEL    = 0.10   # m - |z_W − z_B|  suspension travel
TRUNC_ZS        = 0.30   # m - |z_B|        body displacement
MAX_DISTANCE    = 15   # m - max longitudinal distance per episode (None = unlimited)

#  Observation space bounds (8-dim) 
# [z_B, ż_B, z_W, ż_W, ζ, ζ̇, travel, tyre_defl]
OBS_HIGH = np.array([0.30, 3.0, 0.40, 5.0, 0.15, 7.0, 0.15, 0.05], dtype=np.float32)
OBS_LOW  = -OBS_HIGH

#  Road profile defaults
ROAD_DEFAULTS = {
    'bump_height':      0.1,     # m
    'bump_length':      3.5,     # m
    'bump_x_start':     3.0,     # m - road position where bump begins
    'iso_gd0':          256e-6,  # m³/cycle - ISO 8608 Class C roughness
    'iso_n0':           0.1,     # cycle/m
    'sweep_amplitude':  0.02,    # m
    'episode_duration': 5.0,    # s
}

VEHICLE_SPEED = 5.0   # m/s (36 km/h)

# Speed control 
V_TAU        = 0.5    # s   
V_MAX        = 20.0   # m/s 
V_MIN        = 2.0    # m/s 
V_BRAKE_LEAD = 2.0    # s   — look-ahead time before bump centre to start braking

# Reward — speed terms
W_TIME      = 0.5   # [1/step] — flat per-step time penalty (encourages speed)
W_SPEED_ERR = 0.3   # [dimensionless] — penalty for deviation from v_ref

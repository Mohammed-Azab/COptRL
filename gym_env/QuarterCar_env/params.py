import numpy as np

# ── Quarter-Car Physical Parameters ───────────────────────────────────────────

# Masses
m_B = 465.7   # kg - sprung mass (car body)
m_W = 50.4    # kg - unsprung mass (wheel assembly)

# Tire (linear spring–damper)
c_T = 500.0       # N·s/m - tire damping
k_T = 262_200.0   # N/m   - tire stiffness

# Suspension spring
k_S = 27_922.0    # N/m

# Static equilibrium deflection: Δz_S,stat = m_B·g / k_S
dz_S_stat = m_B * 9.81 / k_S   # ≈ 0.1636 m - exponent reference length in bumpstop

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

# ── Actuator ───────────────────────────────────────────────────────────────────
F_MAX = 10_000.0   # N - active suspension force limit

# ── Control loop timing ────────────────────────────────────────────────────────
DT            = 0.02          # s - control step (50 Hz)
DT_SIM        = 1e-3          # s - RK4 sub-step (1 kHz)
N_SUB         = int(DT / DT_SIM)
EPISODE_STEPS = 500           # steps → 10 s per episode

# ── Truncation thresholds ──────────────────────────────────────────────────────
TRUNC_TRAVEL = 0.10   # m - |z_W − z_B|  suspension travel
TRUNC_ZS     = 0.30   # m - |z_B|        body displacement

# ── Observation space bounds (8-dim) ──────────────────────────────────────────
# [z_B, ż_B, z_W, ż_W, ζ, ζ̇, travel, tyre_defl]
OBS_HIGH = np.array([0.30, 3.0, 0.40, 5.0, 0.15, 7.0, 0.15, 0.05], dtype=np.float32)
OBS_LOW  = -OBS_HIGH

# ── Road profile defaults ──────────────────────────────────────────────────────
ROAD_DEFAULTS = {
    'bump_height':      0.4,     # m
    'bump_length':      3.5,     # m
    'bump_x_start':     0.0,     # m - road position where bump begins
    'iso_gd0':          256e-6,  # m³/cycle - ISO 8608 Class C roughness
    'iso_n0':           0.1,     # cycle/m
    'sweep_amplitude':  0.02,    # m
    'episode_duration': 10.0,    # s
}

VEHICLE_SPEED = 5.0   # m/s (36 km/h)

# ── Render defaults ────────────────────────────────────────────────────────────
RENDER_Y_SCALE   = 5      # vertical exaggeration applied to all z-deflections
RENDER_HIST_SECS = 5.0     # seconds of rolling history in the time-series panel

# Set to False to hide the time-series panel and use full width for the schematic.
RENDER_SHOW_TIMESERIES = True
# Number of time-series subplots shown (1-4): z, z_ddot, F_D, speed.
RENDER_N_TIMESERIES    = 4

# Schematic layout - heights in draw-space (metres, before y-scale is applied).
# z_B and z_W are measured from static equilibrium (= 0), so only deflections
# get multiplied by RENDER_Y_SCALE; these nominal offsets stay fixed.
RENDER_Y_W_NOM = 2.0    # wheel-centre draw height at equilibrium
RENDER_Y_B_NOM = 4.0    # body-centre draw height at equilibrium
RENDER_H_MW    = 0.45   # m_W block height
RENDER_W_MW    = 1.75   # m_W block width
RENDER_H_MB    = 0.45   # m_B block height
RENDER_W_MB    = 1.75  # m_B block width  (wider than m_W)
RENDER_XLIM    = ( -3.0,  15.0)   # x-axis: metres relative to car
RENDER_YLIM    = ( -2.0,   8.5)   # y-axis: draw units
RENDER_ROAD_HALF = 15.0   # road sampled ±this distance from car (m)
RENDER_ROAD_N    = 300    # number of road sample points
RENDER_GROUND_Y  = 1.5    # draw-space offset: shifts ground line + both masses up together

# ── Render appearance — colours ───────────────────────────────────────────────
RENDER_C_MB     = '#f5c842'   # sprung mass body (golden yellow)
RENDER_C_MW     = '#4a86c8'   # unsprung mass (steel blue)
RENDER_C_SPRING = '#e05a1c'   # spring coils (orange-red)
RENDER_C_DAMPER = '#4a86c8'   # damper cylinder / piston (steel blue)
RENDER_C_ROAD   = '#aaaaaa'   # road profile line (light gray)
RENDER_C_GROUND = '#222222'   # ground symbol and contact stem (near-black)

# ── Render appearance — spring geometry ──────────────────────────────────────
RENDER_SP_X = -0.42   # spring centre x in draw-space
RENDER_SP_W =  0.18   # coil half-amplitude (zigzag width)
RENDER_SP_N =  8      # number of zigzag coil pairs

# ── Render appearance — damper geometry ──────────────────────────────────────
# Piston-cylinder style: open-top cylinder (⊔) linked to lower mass via a short
# rod (RENDER_DA_LOWER_STEM); piston linked to upper mass via upper rod.
RENDER_DA_X          =  0.6   # damper centre x in draw-space
RENDER_DA_W          =  0.5   # cylinder full width (wall-to-wall), draw-space
RENDER_DA_PIST_H     =  0.3   # piston height, draw-space (fixed, not scaled)
RENDER_DA_PIST_FRAC  =  0.48  # piston_top = y_lower + gap * this  (0→bottom, 1→top)
RENDER_DA_LOWER_STEM =  0.20  # rod length from lower mass to cylinder base

# Fixed cylinder heights sized from nominal mass-to-mass gaps:
#   susp nominal gap = Y_B_NOM − H_MB/2 − Y_W_NOM − H_MW/2 = 4.0−0.225−2.0−0.175 = 1.6
#   tire nominal gap = Y_W_NOM − H_MW/2                     = 2.0−0.175            = 1.825
RENDER_DA_CYL_H_SUSP = 0.88   # suspension cylinder height (≈ 1.6 × 0.55)
RENDER_DA_CYL_H_TIRE = 0.8   # tire cylinder height       (≈ 1.825 × 0.55)

# ── Render appearance — contact geometry ─────────────────────────────────────
RENDER_CONTACT_STEM = 0   # short stem length above ground line to contact dot
Y_LINE_OFFSET = 0.7   # vertical offset to shift ground line and contact point up together

import numpy as np

# Physical parameters of the quarter-car 2-DOF model
PHYSICS = {
    'm_s': 317.5,    # sprung mass [kg]
    'm_u': 45.4,     # unsprung mass [kg]
    'k_s': 22000.0,  # suspension spring stiffness [N/m]
    'c_s': 1500.0,   # suspension damping [N·s/m]
    'k_t': 192000.0, # tyre stiffness [N/m]
}

# Actuator limit
F_MAX = 10_000.0  # N

# Control loop timing
DT            = 0.02  # s — control step (50 Hz)
EPISODE_STEPS = 500   # steps = 10 s at 50 Hz

# Truncation thresholds — episode ends early if exceeded
TRUNC_TRAVEL = 0.10  # m — |z_s - z_u|
TRUNC_ZS     = 0.30  # m — |z_s|

# Observation space bounds (8-dim)
OBS_HIGH = np.array([0.5, 5.0, 0.5, 5.0, 0.2, 2.0, 0.15, 0.1], dtype=np.float32)
OBS_LOW  = -OBS_HIGH

# Road profile default parameters
ROAD_DEFAULTS = {
    'bump_height':      0.1,    # m
    'bump_length':      0.5,    # m
    'iso_gd0':          256e-6, # m³/cycle — ISO 8608 Class C road roughness
    'iso_n0':           0.1,    # cycle/m — reference spatial frequency
    'sweep_amplitude':  0.02,   # m
    'episode_duration': 10.0,   # s
}

VEHICLE_SPEED = 10.0  # m/s (36 km/h)
MAX_PISODE_STEPS = 1000

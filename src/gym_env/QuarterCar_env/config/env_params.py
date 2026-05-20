import numpy as np

from QuarterCar_env.config.config_manager import _load_yaml

_cfg = _load_yaml("env_params.yaml")

m_B = float(_cfg["m_B"])
m_W = float(_cfg["m_W"])

c_T = float(_cfg["c_T"])
k_T = float(_cfg["k_T"])
k_S = float(_cfg["k_S"])

dz_S_stat = m_B * 9.81 / k_S

_D = float(_cfg["D"])
_A = float(_cfg["A"])

z1 = 2.0 * _D / (1.0 + _A)
d1 = _A * z1

d2 = d1 / 0.25
z2 = z1 / 0.40

v_d = float(_cfg["v_d"])
v_z = float(_cfg["v_z"])

f1_cmp = float(_cfg["f1_cmp"])
f2_cmp = float(_cfg["f2_cmp"])
f1_rbd = float(_cfg["f1_rbd"])
f2_rbd = float(_cfg["f2_rbd"])

dz_cmp = float(_cfg["dz_cmp"])
dz_rbd = float(_cfg["dz_rbd"])

F_ks_nlin_max = float(_cfg["F_ks_nlin_max"])

PHYSICS = {
    "m_B": m_B,
    "m_W": m_W,
    "c_T": c_T,
    "k_T": k_T,
    "k_S": k_S,
    "dz_S_stat": dz_S_stat,
    "d1": d1,
    "z1": z1,
    "d2": d2,
    "z2": z2,
    "v_d": v_d,
    "v_z": v_z,
    "f1_cmp": f1_cmp,
    "f2_cmp": f2_cmp,
    "f1_rbd": f1_rbd,
    "f2_rbd": f2_rbd,
    "dz_cmp": dz_cmp,
    "dz_rbd": dz_rbd,
    "F_ks_nlin_max": F_ks_nlin_max,
}

DT = float(_cfg["DT"])
DT_SIM = float(_cfg["DT_SIM"])
N_SUB = int(DT / DT_SIM)
EPISODE_STEPS = int(_cfg["EPISODE_STEPS"])

TRUNC_TRAVEL = float(_cfg["TRUNC_TRAVEL"])
TRUNC_ZS = float(_cfg["TRUNC_ZS"])
MAX_DISTANCE = float(_cfg["MAX_DISTANCE"])

OBS_HIGH = np.array(_cfg["OBS_HIGH"], dtype=np.float32)
OBS_LOW = -OBS_HIGH


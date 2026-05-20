from QuarterCar_env.config.config_manager import _load_yaml


_cfg = _load_yaml("render_params.yaml")

RENDER_Y_SCALE = int(_cfg["RENDER_Y_SCALE"])
RENDER_HIST_SECS = float(_cfg["RENDER_HIST_SECS"])
RENDER_SHOW_TS = bool(_cfg.get("RENDER_SHOW_TS", False))
RENDER_TS_Z = bool(_cfg.get("RENDER_TS_Z", True))
RENDER_TS_Z_DDOT = bool(_cfg.get("RENDER_TS_Z_DDOT", True))
RENDER_TS_SPEED = bool(_cfg.get("RENDER_TS_SPEED", True))

RENDER_FREEZE_EPISODE = bool(_cfg.get("RENDER_FREEZE_EPISODE", False))

RENDER_Y_W_NOM = float(_cfg["RENDER_Y_W_NOM"])
RENDER_Y_B_NOM = float(_cfg["RENDER_Y_B_NOM"])
RENDER_H_MW = float(_cfg["RENDER_H_MW"])
RENDER_W_MW = float(_cfg["RENDER_W_MW"])
RENDER_H_MB = float(_cfg["RENDER_H_MB"])
RENDER_W_MB = float(_cfg["RENDER_W_MB"])

RENDER_XLIM = tuple(_cfg["RENDER_XLIM"])
RENDER_YLIM = tuple(_cfg["RENDER_YLIM"])

RENDER_ROAD_HALF = float(_cfg["RENDER_ROAD_HALF"])
RENDER_ROAD_N = int(_cfg["RENDER_ROAD_N"])
RENDER_GROUND_Y = float(_cfg["RENDER_GROUND_Y"])

RENDER_C_MB = str(_cfg["RENDER_C_MB"])
RENDER_C_MW = str(_cfg["RENDER_C_MW"])
RENDER_C_SPRING = str(_cfg["RENDER_C_SPRING"])
RENDER_C_DAMPER = str(_cfg["RENDER_C_DAMPER"])
RENDER_C_ROAD = str(_cfg["RENDER_C_ROAD"])
RENDER_C_GROUND = str(_cfg["RENDER_C_GROUND"])

RENDER_SP_X = float(_cfg["RENDER_SP_X"])
RENDER_SP_W = float(_cfg["RENDER_SP_W"])
RENDER_SP_N = int(_cfg["RENDER_SP_N"])

RENDER_DA_X = float(_cfg["RENDER_DA_X"])
RENDER_DA_W = float(_cfg["RENDER_DA_W"])
RENDER_DA_PIST_H = float(_cfg["RENDER_DA_PIST_H"])
RENDER_DA_PIST_FRAC = float(_cfg["RENDER_DA_PIST_FRAC"])
RENDER_DA_LOWER_STEM = float(_cfg["RENDER_DA_LOWER_STEM"])
RENDER_DA_CYL_H_SUSP = float(_cfg["RENDER_DA_CYL_H_SUSP"])
RENDER_DA_CYL_H_TIRE = float(_cfg["RENDER_DA_CYL_H_TIRE"])

RENDER_CONTACT_STEM = float(_cfg["RENDER_CONTACT_STEM"])
Y_LINE_OFFSET = float(_cfg["Y_LINE_OFFSET"])

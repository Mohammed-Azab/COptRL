from QuarterCar_env.config.config_manager import _load_yaml

_cfg = _load_yaml("road_params.yaml")

ROAD_DEFAULTS = dict(_cfg["defaults"])
VEHICLE_SPEED = float(_cfg["vehicle_speed"])
V_BRAKE_LEAD  = float(_cfg["v_brake_lead"])

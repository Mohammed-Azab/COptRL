from QuarterCar_env.config.config_manager import _load_yaml

_cfg = _load_yaml("road_params.yaml")

VEHICLE_SPEED = float(_cfg["vehicle_speed"]) / 3.6   # config in km/h → m/s

MULTI_BUMP_CONFIG = {
    "dis_mode":      _cfg["dis_mode"],
    "num_bumps":     int(_cfg["num_bumps"]),
    "bump_sequence": [int(x) for x in _cfg["bump_sequence"]],
    "custom_dis":    [float(x) for x in _cfg["custom_dis"]],
    "constant_dis":  float(_cfg["constant_dis"]),
    "bump_x_start":  float(_cfg["bump_x_start"]),
    "bump_types":    {int(k): dict(v) for k, v in _cfg["bump_types"].items()},
}

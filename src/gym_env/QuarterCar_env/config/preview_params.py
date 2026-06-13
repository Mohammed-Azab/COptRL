from dataclasses import dataclass
from QuarterCar_env.config.config_manager import _load_yaml


@dataclass(frozen=True)
class PreviewConfig:
    preview_distance:    float = 60.0   # m, lookahead horizon
    h_clip:              float = 0.15   # m, height normalisation clip
    n_peaks:             int   = 3
    peak_height_min:     float = 0.01   # m
    peak_distance_min_m: float = 0.5    # m, min spacing between detected peaks

    noise_active:        bool  = False
    noise_height_std:    float = 0.005
    noise_distance_std:  float = 0.5
    noise_width_std:     float = 0.05

    pt1_tau:             float = 0.05   # s, PT1 filter time constant

    # when True: slot0 = dist/preview_distance, slot2 = width/preview_distance
    # when False: slot0 = t2r/T_MAX,            slot2 = crossing-freq/_FREQ_MAX
    use_dist_obs:        bool  = False


def load_preview_config() -> PreviewConfig:
    try:
        cfg = _load_yaml("preview_params.yaml")
    except FileNotFoundError:
        return PreviewConfig()

    return PreviewConfig(
        preview_distance    = float(cfg.get("preview_distance",    60.0)),
        h_clip              = float(cfg.get("h_clip",              0.15)),
        n_peaks             = int(  cfg.get("n_peaks",             3)),
        peak_height_min     = float(cfg.get("peak_height_min",     0.01)),
        peak_distance_min_m = float(cfg.get("peak_distance_min_m", 0.5)),
        noise_active        = bool( cfg.get("noise_active",        False)),
        noise_height_std    = float(cfg.get("noise_height_std",    0.005)),
        noise_distance_std  = float(cfg.get("noise_distance_std",  0.5)),
        noise_width_std     = float(cfg.get("noise_width_std",     0.05)),
        pt1_tau             = float(cfg.get("pt1_tau",             0.05)),
        use_dist_obs        = bool( cfg.get("use_dist_obs",        False)),
    )

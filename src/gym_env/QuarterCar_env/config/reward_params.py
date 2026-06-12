from dataclasses import dataclass
from QuarterCar_env.config.config_manager import _load_yaml


@dataclass(frozen=True)
class RewardConfig:
    # Mandl weights (g-normalised)
    Q_zBddot:     float = 50.0
    Q_zWddot:     float =  0.5
    Q_a:          float =  1.0
    Q_v:          float =  1.0
    Q_step:       float =  0.1

    # COptRL additions
    w_jerk:          float = 0.4
    w_action_smooth: float = 0.1

    # normalization
    g:               float = 9.81
    reward_heave_clip: float = 8.0
    reward_wheel_clip: float = 60.0

    # velocity (stored internally in m/s; config files use km/h)
    v_limit:    float = 13.9   # m/s, soft speed cap
    a_max:      float =  5.0   # m/s²
    v_min:      float =  2.0   # m/s
    v_init_low: float =  6.94  # m/s (~25 km/h), lower bound for episode v_init sampling

    # longitudinal comfort / filter
    a_comfort:          float = 2.0
    accel_filter_alpha: float = 0.8
    accel_clip:         float = 9.81
    reward_accel_clip:  float = 4.0

    # jerk
    j_max:             float = 2.0
    jerk_filter_alpha: float = 0.8
    jerk_clip:         float = 12.0
    reward_jerk_clip:  float = 4.0

    # terminal
    terminal_bonus:   float = 100.0
    terminal_penalty: float = -100.0
    a_limit:          float =   5.0

    # observation preview
    preview_distance:    float = 60.0
    h_clip:              float = 0.15
    n_peaks:             int   = 3
    peak_height_min:     float = 0.01
    peak_distance_min_m: float = 0.5
    noise_active:        bool  = False
    noise_height_std:    float = 0.005
    noise_distance_std:  float = 0.5
    noise_width_std:     float = 0.05
    pt1_tau:             float = 0.05


def load_reward_config() -> RewardConfig:
    try:
        cfg = _load_yaml("reward_params.yaml")
    except FileNotFoundError:
        return RewardConfig()

    w  = cfg.get("weights",      {})
    vt = cfg.get("vertical",     {})
    v  = cfg.get("velocity",     {})
    c  = cfg.get("comfort",      {})
    j  = cfg.get("jerk",         {})
    t  = cfg.get("terminal",     {})
    o  = cfg.get("observations", {})

    return RewardConfig(
        Q_zBddot        = float(w.get("Q_zBddot",       50.0)),
        Q_zWddot        = float(w.get("Q_zWddot",        0.5)),
        Q_a             = float(w.get("Q_a",             1.0)),
        Q_v             = float(w.get("Q_v",             1.0)),
        Q_step          = float(w.get("Q_step",          0.1)),
        w_jerk          = float(w.get("w_jerk",          0.4)),
        w_action_smooth = float(w.get("w_action_smooth", 0.1)),

        g                = float(vt.get("g",                 9.81)),
        reward_heave_clip = float(vt.get("reward_heave_clip", 8.0)),
        reward_wheel_clip = float(vt.get("reward_wheel_clip", 60.0)),

        v_limit    = float(v.get("v_limit",    50.0)) / 3.6,
        a_max      = float(v.get("a_max",       5.0)),
        v_min      = float(v.get("v_min",       7.2)) / 3.6,
        v_init_low = float(v.get("v_init_low", 25.0)) / 3.6,

        a_comfort          = float(c.get("a_comfort",          2.0)),
        accel_filter_alpha = float(c.get("accel_filter_alpha", 0.8)),
        accel_clip         = float(c.get("accel_clip",         9.81)),
        reward_accel_clip  = float(c.get("reward_accel_clip",  4.0)),

        j_max             = float(j.get("j_max",             2.0)),
        jerk_filter_alpha = float(j.get("jerk_filter_alpha",  0.8)),
        jerk_clip         = float(j.get("jerk_clip",         12.0)),
        reward_jerk_clip  = float(j.get("reward_jerk_clip",   4.0)),

        terminal_bonus   = float(t.get("terminal_bonus",    100.0)),
        terminal_penalty = float(t.get("terminal_penalty", -100.0)),
        a_limit          = float(t.get("a_limit",            5.0)),

        preview_distance    = float(o.get("preview_distance",    60.0)),
        h_clip              = float(o.get("h_clip",              0.15)),
        n_peaks             = int(  o.get("n_peaks",             3)),
        peak_height_min     = float(o.get("peak_height_min",     0.01)),
        peak_distance_min_m = float(o.get("peak_distance_min_m", 0.5)),
        noise_active        = bool( o.get("noise_active",        False)),
        noise_height_std    = float(o.get("noise_height_std",    0.005)),
        noise_distance_std  = float(o.get("noise_distance_std",  0.5)),
        noise_width_std     = float(o.get("noise_width_std",     0.05)),
        pt1_tau             = float(o.get("pt1_tau",             0.05)),
    )

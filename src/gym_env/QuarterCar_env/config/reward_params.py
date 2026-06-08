from dataclasses import dataclass
from QuarterCar_env.config.config_manager import _load_yaml


@dataclass(frozen=True)
class RewardConfig:
    # weights — longitudinal
    w_tracking:      float = 0.5
    w_accel:         float = 0.8
    w_jerk:          float = 0.3
    w_action_smooth: float = 0.2

    # weights — vertical
    w_heave:     float = 0.8
    w_wheel:     float = 0.3
    a_B_comfort: float = 0.5
    a_W_comfort: float = 30.0
    reward_heave_clip: float = 1.0
    reward_wheel_clip: float = 60.0
    enable_heave:       bool = True
    enable_wheel:       bool = True
    enable_vel_scaling: bool = True

    # enable flags — longitudinal
    enable_tracking:      bool = True
    enable_accel:         bool = True
    enable_jerk:          bool = True
    enable_action_smooth: bool = True
    enable_progress:      bool = True

    # progress reward weight (positive: encourages forward movement)
    w_progress: float = 0.2

    # bump-crossing reward: positive one-shot reward each time a bump end is cleared
    w_bump_cross: float = 5.0

    # velocity (stored internally in m/s; config files use km/h)
    v_max: float = 20.0   # m/s
    a_max: float = 5.0    # m/s²
    v_min: float = 2.0    # m/s

    # longitudinal comfort / filter
    a_comfort:          float = 2.0
    accel_filter_alpha: float = 0.8
    accel_clip:         float = 8.0
    reward_accel_clip:  float = 4.0

    # jerk
    j_max:             float = 2.0
    jerk_filter_alpha: float = 0.8
    jerk_clip:         float = 12.0
    reward_jerk_clip:  float = 4.0

    # terminal
    terminal_bonus:   float = 100.0
    terminal_penalty: float = -100.0
    a_limit:          float = 1.0

    # observation — preview
    preview_distance:    float = 20.0
    h_clip:              float = 0.15
    n_peaks:             int   = 3
    peak_height_min:     float = 0.01
    peak_distance_min_m: float = 0.5
    noise_active:        bool  = True
    noise_height_std:    float = 0.005
    noise_distance_std:  float = 0.5
    noise_width_std:     float = 0.05
    pt1_tau:             float = 0.2


def load_reward_config() -> RewardConfig:
    try:
        cfg = _load_yaml("reward_params.yaml")
    except FileNotFoundError:
        return RewardConfig()

    w  = cfg.get("weights",      {})
    vt = cfg.get("vertical",     {})
    e  = cfg.get("enable",       {})
    v  = cfg.get("velocity",     {})
    c  = cfg.get("comfort",      {})
    j  = cfg.get("jerk",         {})
    t  = cfg.get("terminal",     {})
    o  = cfg.get("observations", {})

    return RewardConfig(
        w_tracking      = float(w.get("w_tracking",      0.5)),
        w_accel         = float(w.get("w_accel",         0.8)),
        w_jerk          = float(w.get("w_jerk",          0.3)),
        w_action_smooth = float(w.get("w_action_smooth", 0.2)),

        w_heave          = float(vt.get("w_heave",          0.8)),
        w_wheel          = float(vt.get("w_wheel",          0.3)),
        a_B_comfort      = float(vt.get("a_B_comfort",      0.5)),
        a_W_comfort      = float(vt.get("a_W_comfort",     30.0)),
        reward_heave_clip = float(vt.get("reward_heave_clip", 1.0)),
        reward_wheel_clip = float(vt.get("reward_wheel_clip", 60.0)),
        enable_heave         = bool(vt.get("enable_heave",         True)),
        enable_wheel         = bool(vt.get("enable_wheel",         True)),
        enable_vel_scaling   = bool(vt.get("enable_vel_scaling",   True)),

        enable_tracking      = bool(e.get("tracking",      True)),
        enable_accel         = bool(e.get("accel",         True)),
        enable_jerk          = bool(e.get("jerk",          True)),
        enable_action_smooth = bool(e.get("action_smooth", True)),
        enable_progress      = bool(e.get("progress",      True)),
        w_progress           = float(e.get("w_progress",   0.2)),
        w_bump_cross         = float(e.get("w_bump_cross", 5.0)),

        v_max = float(v.get("v_max", 72.0)) / 3.6,   # config in km/h → m/s
        a_max = float(v.get("a_max",  5.0)),
        v_min = float(v.get("v_min",  7.2)) / 3.6,   # config in km/h → m/s

        a_comfort          = float(c.get("a_comfort",          2.0)),
        accel_filter_alpha = float(c.get("accel_filter_alpha", 0.8)),
        accel_clip         = float(c.get("accel_clip",         8.0)),
        reward_accel_clip  = float(c.get("reward_accel_clip",  4.0)),

        j_max             = float(j.get("j_max",             2.0)),
        jerk_filter_alpha = float(j.get("jerk_filter_alpha",  0.8)),
        jerk_clip         = float(j.get("jerk_clip",         12.0)),
        reward_jerk_clip  = float(j.get("reward_jerk_clip",   4.0)),

        terminal_bonus   = float(t.get("terminal_bonus",    100.0)),
        terminal_penalty = float(t.get("terminal_penalty", -100.0)),
        a_limit          = float(t.get("a_limit",            1.0)),

        preview_distance    = float(o.get("preview_distance",    20.0)),
        h_clip              = float(o.get("h_clip",              0.15)),
        n_peaks             = int(  o.get("n_peaks",             3)),
        peak_height_min     = float(o.get("peak_height_min",     0.01)),
        peak_distance_min_m = float(o.get("peak_distance_min_m", 0.5)),
        noise_active        = bool( o.get("noise_active",        True)),
        noise_height_std    = float(o.get("noise_height_std",    0.005)),
        noise_distance_std  = float(o.get("noise_distance_std",  0.5)),
        noise_width_std     = float(o.get("noise_width_std",     0.05)),
        pt1_tau             = float(o.get("pt1_tau",             0.2)),
    )

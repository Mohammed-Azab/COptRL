from dataclasses import dataclass
from QuarterCar_env.config.config_manager import _load_yaml


@dataclass(frozen=True)
class RewardConfig:
    # --- weights ---
    w_comfort_bonus: float = 0.8   # positive per-step bonus for smooth riding
    w_tracking:      float = 1.0
    w_accel:         float = 0.5
    w_jerk:          float = 0.3
    w_action_smooth: float = 0.2
    w_curve:         float = 0.0

    # --- per-term enable flags ---
    enable_comfort_bonus: bool = True
    enable_tracking:      bool = True
    enable_accel:         bool = True
    enable_jerk:          bool = True
    enable_action_smooth: bool = True
    enable_curve:         bool = True

    # --- velocity ---
    v_max:             float = 20.0
    a_max:             float = 5.0
    target_speed_mode: str   = "constant"   # constant | curvature_aware | external
    a_lat_comfort:     float = 2.0
    min_curve_speed:   float = 2.0
    max_curve_speed:   float = 20.0

    # --- longitudinal comfort ---
    a_comfort:          float = 3.0
    accel_filter_alpha: float = 0.8
    accel_clip:         float = 15.0  # observation clipping bound (wide, for numerical safety)
    reward_accel_clip:  float = 6.0   # reward-only clip = 2 × a_comfort; keeps worst-case bounded

    # --- jerk ---
    j_max:             float = 10.0
    jerk_filter_alpha: float = 0.8
    jerk_clip:         float = 50.0  # observation clipping bound
    reward_jerk_clip:  float = 20.0  # reward-only clip = 2 × j_max; keeps worst-case bounded

    # --- curvature ---
    a_lat_max:      float = 4.0
    curvature_clip: float = 0.5

    # --- terminal ---
    terminal_bonus:   float = 100.0
    terminal_penalty: float = -100.0
    a_limit:          float = 10.0   # RMS threshold for terminal bonus / comfort_score

    # --- observation toggles (read once at env __init__) ---
    obs_enable_accel:             bool = True
    obs_enable_jerk:              bool = True
    obs_enable_prev_action:       bool = True
    obs_enable_curvature:         bool = True
    obs_enable_curvature_preview: bool = False

    # --- road preview ---
    obs_enable_preview:  bool  = True   
    preview_distance:    float = 20.0   # m [lookahead horizon]
    n_preview_points:    int   = 10     # number of height samples in the preview window
    preview_height_clip: float = 0.15   # m — clip before normalising (matches OBS_HIGH[4])


def load_reward_config() -> RewardConfig:
    """Load RewardConfig from reward_params.yaml. Falls back to dataclass defaults on error."""
    try:
        cfg = _load_yaml("reward_params.yaml")
    except FileNotFoundError:
        return RewardConfig()

    w = cfg.get("weights",      {})
    e = cfg.get("enable",       {})
    v = cfg.get("velocity",     {})
    c = cfg.get("comfort",      {})
    j = cfg.get("jerk",         {})
    k = cfg.get("curvature",    {})
    t = cfg.get("terminal",     {})
    o = cfg.get("observations", {})

    return RewardConfig(
        w_comfort_bonus = float(w.get("w_comfort_bonus", 0.8)),
        w_tracking      = float(w.get("w_tracking",      1.0)),
        w_accel         = float(w.get("w_accel",         0.5)),
        w_jerk          = float(w.get("w_jerk",          0.3)),
        w_action_smooth = float(w.get("w_action_smooth", 0.2)),
        w_curve         = float(w.get("w_curve",         0.0)),

        enable_comfort_bonus = bool(e.get("comfort_bonus",  True)),
        enable_tracking      = bool(e.get("tracking",      True)),
        enable_accel         = bool(e.get("accel",         True)),
        enable_jerk          = bool(e.get("jerk",          True)),
        enable_action_smooth = bool(e.get("action_smooth", True)),
        enable_curve         = bool(e.get("curve",         True)),

        v_max             = float(v.get("v_max",            20.0)),
        a_max             = float(v.get("a_max",             5.0)),
        target_speed_mode = str(  v.get("target_speed_mode", "constant")),
        a_lat_comfort     = float(v.get("a_lat_comfort",      2.0)),
        min_curve_speed   = float(v.get("min_curve_speed",    2.0)),
        max_curve_speed   = float(v.get("max_curve_speed",   20.0)),

        a_comfort          = float(c.get("a_comfort",          3.0)),
        accel_filter_alpha = float(c.get("accel_filter_alpha", 0.8)),
        accel_clip         = float(c.get("accel_clip",        15.0)),
        reward_accel_clip  = float(c.get("reward_accel_clip",  6.0)),

        j_max             = float(j.get("j_max",            10.0)),
        jerk_filter_alpha = float(j.get("jerk_filter_alpha", 0.8)),
        jerk_clip         = float(j.get("jerk_clip",        50.0)),
        reward_jerk_clip  = float(j.get("reward_jerk_clip", 20.0)),

        a_lat_max      = float(k.get("a_lat_max",      4.0)),
        curvature_clip = float(k.get("curvature_clip", 0.5)),

        terminal_bonus   = float(t.get("terminal_bonus",    100.0)),
        terminal_penalty = float(t.get("terminal_penalty", -100.0)),
        a_limit          = float(t.get("a_limit",           10.0)),

        obs_enable_accel             = bool( o.get("obs_enable_accel",             True)),
        obs_enable_jerk              = bool( o.get("obs_enable_jerk",              True)),
        obs_enable_prev_action       = bool( o.get("obs_enable_prev_action",       True)),
        obs_enable_curvature         = bool( o.get("obs_enable_curvature",         True)),
        obs_enable_curvature_preview = bool( o.get("obs_enable_curvature_preview", False)),

        obs_enable_preview  = bool( o.get("obs_enable_preview",  True)),
        preview_distance    = float(o.get("preview_distance",    20.0)),
        n_preview_points    = int(  o.get("n_preview_points",    10)),
        preview_height_clip = float(o.get("preview_height_clip", 0.15)),
    )

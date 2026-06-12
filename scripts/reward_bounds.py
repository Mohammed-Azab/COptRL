import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src' / 'gym_env'))
sys.path.insert(0, str(_ROOT / 'src'))

from QuarterCar_env.config.reward_params import load_reward_config
from QuarterCar_env.config.env_params import EPISODE_STEPS, DT
from QuarterCar_env.reward.utils import reward_bounds

cfg = load_reward_config()
b   = reward_bounds(cfg, EPISODE_STEPS)

print(f"per-step : [{b['per_step_min']:+.4f}, {b['per_step_max']:+.4f}]")
print(f"episode  : [{b['episode_min']:+.1f}, {b['episode_max']:+.1f}]  ({EPISODE_STEPS} steps, DT={DT})")
print()
print(f"terminal_bonus   : {cfg.terminal_bonus:+.1f}")
print(f"terminal_penalty : {cfg.terminal_penalty:+.1f}")
print(f"j_time max extra : {cfg.terminal_bonus:+.1f}  (completion at t=0)")
print(f"best episode     : {b['per_step_max'] * EPISODE_STEPS + cfg.terminal_bonus * 2:+.1f}  (perfect steps + both terminal bonuses)")

# QuarterCar_env

A custom [Gymnasium](https://gymnasium.farama.org) environment for training reinforcement learning agents on an active vehicle suspension control problem.

Scaffolded with the [Gymnasium environment template](https://github.com/Farama-Foundation/gymnasium-env-template) but fully replaced with a quarter-car physics simulation instead of the template examples.

---

## Environment

**`QuarterCar_env/QuarterCar`** — 2-DOF quarter-car model with an active actuator between sprung and unsprung masses.

| | |
|---|---|
| Observation space | `Box(8,)` float32 |
| Action space | `Box(1,)` float32 in `[-1, 1]` |
| Control frequency | 50 Hz (dt = 0.02 s) |
| Episode length | 500 steps (10 s) |

### Observation vector

| idx | Symbol | Description | Clip |
|-----|--------|-------------|------|
| 0 | z_s | Sprung mass displacement [m] | ±0.5 |
| 1 | ż_s | Sprung mass velocity [m/s] | ±5 |
| 2 | z_u | Unsprung mass displacement [m] | ±0.5 |
| 3 | ż_u | Unsprung mass velocity [m/s] | ±5 |
| 4 | z_r | Road height [m] | ±0.2 |
| 5 | ż_r | Road height rate [m/s] | ±2 |
| 6 | z_s − z_u | Suspension travel [m] | ±0.15 |
| 7 | z_u − z_r | Tyre deflection [m] | ±0.1 |

### Action

`action[0] ∈ [-1, 1]` scaled to actuator force `F_act = action[0] × 10 000 N`.

### Road profiles

| Profile | Description |
|---------|-------------|
| `speed_bump` | Single sinusoidal bump |
| `iso_8608_class_c` | Stochastic rough road (ISO 8608 Class C PSD) |
| `sine_sweep` | Frequency sweep 0.5 – 20 Hz |
| `flat` | No road disturbance |

### Reward

Penalty-based: comfort (sprung acceleration), suspension travel, tyre deflection, and actuator effort. Terminal bonus on clean episode completion.

---

## Package layout

```
gym_env/
├── pyproject.toml
└── QuarterCar_env/
    ├── __init__.py          # gymnasium registration
    ├── params.py            # physical and RL constants
    ├── ode_model.py         # 2-DOF ODE integrator (scipy RK45)
    ├── reward.py            # reward function and config
    ├── road_generator.py    # road profile generator
    ├── envs/
    │   ├── __init__.py
    │   └── quarter_car_env.py   # QuarterCarEnv (gym.Env)
    └── wrappers/
        ├── __init__.py
        ├── normalize_observation.py
        ├── action_repeat.py
        ├── reward_scaler.py
        └── episode_logger.py
```

---

## Installation

```bash
pip install -e gym_env/
```

---

## Quick start

```python
import gymnasium as gym
import QuarterCar_env  # registers the environment

env = gym.make("QuarterCar_env/QuarterCar", road_profile="speed_bump")
obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Development

```bash
pre-commit install   # black + ruff formatting hooks
```

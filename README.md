# COptRL — Comfort Optimizer RL

> *Teach a car to slow down before the bump, not after.*

COptRL is a quarter-car suspension simulation and training framework where a deep RL agent learns **longitudinal speed profiles** that minimise ride discomfort (vertical body and wheel accelerations) while tracking a target velocity. Instead of active suspension, this framing asks: *can the agent plan its speed so the bump never causes discomfort in the first place?*

Built on Gymnasium · Stable-Baselines3 · Numba.

---

## Problem Framing

The agent observes the suspension state and an upcoming peak-detected road preview, then outputs a normalised acceleration command every 20 ms. The quarter-car ODE integrates vertical dynamics passively — the agent's only lever is the longitudinal speed it arrives at any given point.

```
Observation (6 + 3×n_peaks D) ──► PPO Agent ──► u ∈ [-1, 1]
                                                        │
                                             a_cmd = u × a_max  [m/s²]
                                             v_{t+1} = clip(v_t + a_cmd × DT, 0, v_max)
                                                        │
                                             Quarter-car ODE (6-state, RK4)
                                                        │
                                             Reward: (v/v_max) × (vertical + longitudinal)
```

---

## Repository Layout

```
COptRL/
├── config/
│   ├── algo/algo_configs.yaml          PPO / TD3 hyperparameters
│   ├── curriculum/curriculum_params.yaml  3-level curriculum config
│   ├── eval/compare_config.yaml        comparison settings
│   ├── gym_env/env_params.yaml         physics constants, episode length
│   ├── reward/reward_params.yaml       reward weights, vertical terms, preview config
│   ├── road/road_params.yaml           bump layout + random road bounds
│   └── scenarios/                      named experiment scenarios (JSON)
│
├── src/
│   ├── gym_env/QuarterCar_env/
│   │   ├── core/ode_model.py           6-state ODE, Numba-jitted RK4
│   │   ├── envs/quarter_car_env.py     Gymnasium Env (6-D obs, step, reset)
│   │   ├── reward/reward.py            r_heave, r_wheel, r_tracking, r_accel, r_jerk, r_smooth
│   │   ├── config/                     RewardConfig, env_params, road_params
│   │   └── wrappers/
│   │       ├── preview.py              PreviewWrapper — peak detection + noise + PT1
│   │       ├── curriculum.py           CurriculumWrapper — staged road difficulty
│   │       ├── episode_logger.py
│   │       ├── normalize_observation.py
│   │       ├── action_repeat.py
│   │       └── reward_scaler.py
│   │
│   ├── road/road_generator.py          speed_bump, flat, recorded profiles + from_random
│   │
│   ├── train/
│   │   ├── train.py                    training entry point (--curriculum flag)
│   │   ├── agent.py                    algorithm registry (PPO / TD3)
│   │   ├── environment.py              make_vec_env / make_eval_vec_env + wrappers
│   │   ├── monitoring.py               callbacks: metrics, checkpoint, sync
│   │   └── seed.py                     global seed helper
│   │
│   ├── eval/
│   │   ├── eval.py                     single-model deep evaluation (--save-plots)
│   │   └── compare.py                  agent vs baseline comparison + plots + JSON
│   │
│   └── tune/
│       ├── hyperparameter_search.py    Optuna HP search
│       ├── search_spaces.py            per-algorithm samplers
│       └── trial.py                    Optuna objective
│
├── tests/                              pytest suite
├── models/                             saved checkpoints (gitignored)
└── logs/                               TensorBoard + Monitor logs (gitignored)
```

---

## The Physics

A **2-DOF quarter-car** with nonlinear suspension (exponential bumpstop, asymmetric sigmoid damper):

```
  m_B · z̈_B = −k_S(z_W−z_B) − F_damp(ż_W−ż_B) − F_bumpstop
  m_W · z̈_W =  k_S(z_W−z_B) + F_damp + F_bumpstop − k_T(z_W−ζ) − c_T(ż_W−ζ̇)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| m_B | 465.7 kg | Sprung mass (body) |
| m_W | 50.4 kg | Unsprung mass (wheel) |
| k_S | 27 922 N/m | Suspension spring |
| k_T | 262 200 N/m | Tyre radial stiffness |
| c_T | 500 N·s/m | Tyre damping |

Integration: RK4 with 20 sub-steps per 20 ms control step (1 ms physics step).

### State Vector

```
x[0] = ζ − z_W    tyre deflection         [m]
x[1] = ż_W        wheel vertical velocity  [m/s]
x[2] = z_W − z_B  suspension travel        [m]
x[3] = ż_B        body vertical velocity   [m/s]
x[4] = v          longitudinal speed       [m/s]   ← driven by agent
x[5] = z_B        body displacement        [m]
```

---

## Observation Space

**Base env** outputs 6 scalars; `PreviewWrapper` appends `3 × n_peaks` peak slots:

| Index | Signal | Notes |
|-------|--------|-------|
| 0 | ζ (road height) | clipped to ±0.15 m |
| 1 | ζ̇ (road velocity) | clipped to ±7.0 m/s |
| 2 | v / v_max | normalised speed [0, 1] |
| 3 | filtered_a / a_comfort | smoothed longitudinal accel |
| 4 | filtered_jerk / j_max | smoothed jerk |
| 5 | prev_action | last command [-1, 1] |
| 6 … 6+3n−1 | peak preview | `[dist/D, h/h_clip, width/D]` × n_peaks, all ∈ [0, 1] |

Missing peaks fill with `[1.0, 0.0, 0.0]` (bump at horizon).

---

## Reward Function

```
R_step = (v / v_max) × (
    w_heave  × r_heave          ← −(z̈_B / a_B_comfort)²
    w_wheel  × r_wheel          ← −(z̈_W / a_W_comfort)²
    w_track  × r_tracking       ← 0 if v in band, else −(err/v_min)²
    w_accel  × r_accel          ← −(filtered_a / a_comfort)²
    w_jerk   × r_jerk           ← −(filtered_jerk / j_max)²
    w_smooth × r_action_smooth  ← −(u_t − u_{t-1})²
)
```

Velocity scaling `(v/v_max)` means low-speed episodes contribute proportionally less — same principle as ba_azab.

**Terminal signal:** `+100` if episode RMS body accel < `a_limit`, else `−100`.

---

## Road Profiles

| Profile | Description |
|---------|-------------|
| `speed_bump` | Versine bump(s) — height, length, and count configurable |
| `flat` | ζ = 0, speed-tracking isolation |
| `recorded` | Load from JSON scenario file |

Per-episode random bump layout via `RoadGenerator.from_random(rng, speed)` — wired into `reset()` by default.

---

## Curriculum

Three staged difficulty levels controlled by `config/curriculum/curriculum_params.yaml`:

| Level | Bumps | Height | Speed |
|-------|-------|--------|-------|
| 0 | 1–2 | 0.05–0.10 m | 4–10 m/s |
| 1 | 1–3 | 0.05–0.15 m | 4–15 m/s |
| 2 | 1–5 | 0.05–0.25 m | 4–20 m/s |

Levels advance based on total env steps. Step thresholds are configurable.

---

## Quickstart

### Install

```bash
git clone https://github.com/Mohammed-Azab/COptRL.git
cd COptRL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/gym_env
```

### Train

```bash
# PPO with curriculum (recommended)
python src/train/train.py --algo PPO --road speed_bump --curriculum

# PPO without curriculum (random road every episode)
python src/train/train.py --algo PPO --road speed_bump

# TD3
python src/train/train.py --algo TD3 --road speed_bump --curriculum

# Resume from checkpoint
python src/train/train.py --algo PPO --resume models/PPO/speed_bump/exp_1/PPO_final.zip
```

Training writes to:
```
models/<ALGO>/<road>/<run_tag>/
    <algo>_final.zip
    vecnormalize.pkl
    best/best_model.zip
    checkpoints/

logs/tensorboard/<run_tag>/
logs/monitor/<run_tag>/
```

```bash
tensorboard --logdir logs/tensorboard
```

### Evaluate

```bash
# Deep single-model eval with plots
python src/eval/eval.py \
    --algo PPO \
    --model_path models/PPO/speed_bump/exp_1/PPO_final.zip \
    --save-plots

# Compare agent vs passive/random baselines
python src/eval/compare.py \
    --algo PPO \
    --model-path models/PPO/speed_bump/exp_1/PPO_final.zip \
    --save-plots
```

### Hyperparameter Search

```bash
python src/tune/hyperparameter_search.py --algo ppo --trials 50 --timesteps 100000
```

---

## Algorithms

| Algorithm | Type | Notes |
|-----------|------|-------|
| **PPO** | On-policy | Default; reward normalisation on; flat MLP policy |
| **TD3** | Off-policy | No reward normalisation |

---

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Key Design Decisions

**Speed planning, not active suspension.** The agent only controls longitudinal speed. Vertical dynamics are passive — making the problem harder but physically meaningful.

**Velocity-scaled reward.** `R *= v/v_max` so low-speed episodes contribute proportionally less to gradient updates.

**Peak-detected preview.** `PreviewWrapper` detects up to N bumps in the lookahead window and encodes each as `[dist, height, width]` with Gaussian noise and a PT1 filter — mimicking an imperfect sensor.

**Curriculum.** `CurriculumWrapper` starts with small slow bumps and expands the difficulty range as training progresses, giving the agent informative gradients from the start.

**Geometry-safe speed.** `RoadGenerator._clamp_speed_to_geometry()` caps vehicle speed to a physics-calibrated limit based on bump height/length ratio, preventing impossible training scenarios.

---

*Python 3.10+ · Stable-Baselines3 · Gymnasium · Numba · SciPy · Matplotlib*

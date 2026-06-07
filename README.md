# COptRL — Comfort-Optimising Reinforcement Learning for Active Speed Planning

## Abstract

This work presents **COptRL**, a reinforcement learning framework that trains an agent to plan longitudinal vehicle speed in a way that minimises passenger discomfort when traversing road disturbances. Rather than relying on active suspension hardware to suppress vibrations in real time, the agent learns a predictive control policy: it observes an upcoming road preview and adjusts speed proactively so that the quarter-car suspension is excited within comfortable limits.

The environment is built on a nonlinear 6-state quarter-car model integrated at 1 ms resolution via a Numba-jitted RK4 solver, wrapped in a Gymnasium-compatible interface. The policy is trained with Proximal Policy Optimisation (PPO) using a staged curriculum, per-episode road randomisation, and a hybrid reward signal that combines vertical body and wheel accelerations (ISO 2631-aligned) with longitudinal tracking and smoothness terms scaled by instantaneous vehicle speed.

---

## 1. Motivation and Problem Framing

Classical active suspension systems act *reactively*: a sensor detects a bump and an actuator generates a counterforce within milliseconds. This demands fast, expensive hardware and still suffers from actuator latency. An alternative — explored here — is *predictive speed planning*: if the vehicle arrives at a speed bump slowly enough, the passive suspension absorbs the disturbance without exceeding comfort thresholds. No actuator is needed; the agent's only control variable is the longitudinal acceleration command.

This framing raises a well-defined RL problem:

> **Given a preview of upcoming road geometry and the current suspension state, find a speed profile that minimises vertical body acceleration while maintaining a target velocity.**

The agent cannot see the internal ODE state directly. It observes road contact signals (ζ, ζ̇), filtered longitudinal dynamics, and a peak-detected preview of bumps ahead. The reward signal is proportional to vehicle speed, so the agent is implicitly penalised for stopping (trivial comfort via zero excitation) and rewarded for fast, smooth traversal.

---

## 2. Environment

### 2.1 Quarter-Car Model

A 2-DOF nonlinear quarter-car with sprung mass *m*_B (body) and unsprung mass *m*_W (wheel):

```
m_B · z̈_B  =  −F_spring(z_W − z_B) − F_damp(ż_W − ż_B)
m_W · z̈_W  =   F_spring + F_damp − k_T(z_W − ζ) − c_T(ż_W − ζ̇)
```

The suspension spring includes an exponential bumpstop progression beyond the linear clearance zone. The damper is modelled with a sigmoid-blended asymmetric piecewise-linear characteristic (different compression and rebound slopes) to capture realistic damper behaviour.

| Parameter | Value | Description |
|-----------|-------|-------------|
| m_B | 465.7 kg | Sprung mass |
| m_W | 50.4 kg | Unsprung mass |
| k_S | 27 922 N/m | Suspension spring stiffness |
| k_T | 262 200 N/m | Tyre radial stiffness |
| c_T | 500 N·s/m | Tyre damping coefficient |

Integration uses **RK4 with 20 sub-steps** per 20 ms control interval (effective physics step: 1 ms). All integration kernels are Numba-jitted; a full 300-step episode runs in under 5 ms on CPU.

**State vector:**

| Index | Variable | Description |
|-------|----------|-------------|
| x[0] | ζ − z_W | Tyre deflection [m] |
| x[1] | ż_W | Wheel vertical velocity [m/s] |
| x[2] | z_W − z_B | Suspension travel [m] |
| x[3] | ż_B | Body vertical velocity [m/s] |
| x[4] | v | Longitudinal speed [m/s] (set by agent) |
| x[5] | z_B | Body displacement [m] |

### 2.2 Road Generator

Three road profiles are supported:

| Profile | Description |
|---------|-------------|
| `speed_bump` | Versine bump(s): `z(x) = (A/2)(1 − cos(2πx/L))` |
| `flat` | Zero excitation; used for speed-tracking baselines |
| `recorded` | Real-world measurement loaded from a JSON scenario file |

Per-episode randomisation is provided by `RoadGenerator.from_random(rng, speed)`, which samples bump count, height, length, and inter-bump gaps from configurable ranges. Vehicle speed is clamped to a geometry-safe limit derived from an empirical calibration of height-to-length ratio vs. maximum safe traversal speed (two calibration points from physical test runs).

### 2.3 Action and Observation Spaces

**Action:** scalar float32 ∈ [−1, 1]. Scaled to `a_cmd = u × a_max` [m/s²]; speed is integrated as `v_{t+1} = clip(v_t + a_cmd × DT, 0, v_max)`.

**Observation:** the base environment outputs 6 scalar features; `PreviewWrapper` appends `3 × n_peaks` peak-encoded features.

| Index | Signal | Normalisation |
|-------|--------|--------------|
| 0 | ζ (road height at wheel) | clipped ±0.15 m |
| 1 | ζ̇ (road velocity at wheel) | clipped ±7.0 m/s |
| 2 | v / v_max | [0, 1] |
| 3 | filtered_a / a_comfort | IIR-smoothed longitudinal accel |
| 4 | filtered_jerk / j_max | IIR-smoothed jerk |
| 5 | prev_action | [-1, 1] |
| 6 … 6+3n−1 | peak preview | `[dist/D, h/h_clip, w/D]` per peak |

Longitudinal signals use a first-order IIR filter (α = 0.8) to reduce noise and provide a smoother gradient signal to the policy.

### 2.4 Peak-Detected Preview

The `PreviewWrapper` samples 200 evenly-spaced height values over the lookahead horizon (default 20 m), runs `scipy.signal.find_peaks` to locate up to *n* bumps, and encodes each detected bump as a 3-tuple:

```
[distance_ahead / D,  height / h_clip,  width / D]
```

Missing peaks (fewer than *n* detected) fill with `[1.0, 0.0, 0.0]`, representing a bump at the horizon with zero height — the neutral state.

Gaussian noise is added to each detected peak, scaled by the peak's distance normalisation (closer bumps have proportionally larger noise), then a first-order PT1 filter (τ = 0.2 s) is applied. This mimics the behaviour of an imperfect real-world sensor and introduces a structured stochastic element that improves policy robustness.

---

## 3. Reward Function

The per-step reward is:

```
R = (v / v_max) × [
      w_heave  · r_heave(z̈_B)       +   w_wheel  · r_wheel(z̈_W)
    + w_track  · r_tracking(v)       +   w_accel  · r_accel(ã_long)
    + w_jerk   · r_jerk(j̃_long)     +   w_smooth · r_smooth(u_t, u_{t-1})
]
```

where individual terms are:

| Term | Formula | Physical meaning |
|------|---------|-----------------|
| r_heave | −(z̈_B / a_B_comfort)² | ISO 2631 body comfort (vertical) |
| r_wheel | −(z̈_W / a_W_comfort)² | Wheel-hop / road holding |
| r_tracking | 0 if v ∈ [v_min, v_ref]; −(err/v_min)² else | Velocity reference tracking |
| r_accel | −(ã_long / a_comfort)² | Longitudinal ride comfort |
| r_jerk | −(j̃_long / j_max)² | Smoothness of acceleration |
| r_smooth | −(u_t − u_{t-1})² | Control continuity |

The velocity scaling factor `(v / v_max)` ensures that low-speed episodes contribute proportionally less to the gradient signal, discouraging the degenerate strategy of stopping before every bump.

A sparse terminal signal of ±100 is applied at episode end based on whether episode RMS body acceleration falls below the comfort threshold `a_limit`.

Default weights: w_heave = 1.0, w_wheel = 0.5, w_track = 0.5, w_accel = 0.8, w_jerk = 0.3, w_smooth = 0.2.

---

## 4. Training

### 4.1 Algorithm

**Proximal Policy Optimisation (PPO)** with a standard `MlpPolicy` [256, 256] for both actor and critic. Reward normalisation is enabled via `VecNormalize`. The flat MLP sees the full observation — including preview slots — without architectural separation, following the approach used in the reference implementation (ba_azab).

### 4.2 Curriculum Learning

A three-level curriculum progressively expands road difficulty:

| Level | Bumps | Height range | Speed range | Unlocked at |
|-------|-------|-------------|-------------|-------------|
| 0 | 1–2 | 0.05–0.10 m | 4–10 m/s | start |
| 1 | 1–3 | 0.05–0.15 m | 4–15 m/s | 200 k steps |
| 2 | 1–5 | 0.05–0.25 m | 4–20 m/s | 500 k steps |

The `CurriculumWrapper` intercepts `reset()` and injects level-appropriate `from_random` parameters via the options dict, advancing level based on total env steps accumulated. All thresholds are configurable in `config/curriculum/curriculum_params.yaml`.

### 4.3 Per-Episode Randomisation

At every episode reset (all levels), bump count, heights, lengths, and inter-bump gaps are independently sampled. Vehicle entry speed is also sampled uniformly within the level's range and then hard-clamped to a geometry-safe limit. This prevents the policy from overfitting to any fixed road layout.

---

## 5. Repository Structure

```
COptRL/
├── config/
│   ├── algo/algo_configs.yaml              PPO / TD3 hyperparameters (SB3 kwargs)
│   ├── curriculum/curriculum_params.yaml   level thresholds and bump bounds
│   ├── eval/compare_config.yaml            comparison evaluation settings
│   ├── gym_env/env_params.yaml             physics constants, DT, episode length
│   ├── reward/reward_params.yaml           weights, vertical terms, preview config
│   ├── road/road_params.yaml               fixed bump layout + random bounds
│   └── scenarios/                          named evaluation scenarios (JSON)
│
├── src/
│   ├── gym_env/QuarterCar_env/
│   │   ├── core/ode_model.py               Numba-jitted RK4, spring + damper kernels
│   │   ├── envs/quarter_car_env.py         Gymnasium Env: 6-D base observation, reset
│   │   ├── reward/
│   │   │   ├── reward.py                   all reward primitives + compute_reward()
│   │   │   └── utils.py                    reward_bounds() for analysis
│   │   ├── config/                         RewardConfig dataclass, YAML loaders
│   │   └── wrappers/
│   │       ├── preview.py                  peak detection, noise, PT1 filter
│   │       ├── curriculum.py               staged difficulty injection
│   │       ├── episode_logger.py           per-episode CSV logging
│   │       ├── normalize_observation.py    running mean/std normalisation
│   │       ├── action_repeat.py            frame-skip wrapper
│   │       └── reward_scaler.py            constant reward scaling
│   │
│   ├── road/road_generator.py              speed_bump / flat / recorded + from_random
│   │
│   ├── train/
│   │   ├── train.py                        entry point (--curriculum, --algo, --road)
│   │   ├── agent.py                        algorithm registry: PPO, TD3
│   │   ├── environment.py                  vec env builder with wrappers
│   │   ├── monitoring.py                   EvalCallback, CheckpointCallback
│   │   └── seed.py                         deterministic seeding helper
│   │
│   ├── eval/
│   │   ├── eval.py                         per-episode deep inspection + --save-plots
│   │   └── compare.py                      agent vs baselines, multi-road, --save-plots
│   │
│   └── tune/
│       ├── hyperparameter_search.py        Optuna study runner
│       ├── search_spaces.py                per-algorithm Optuna samplers
│       └── trial.py                        Optuna objective (train + eval)
│
└── tests/
    ├── test_ode.py         ODE step returns (z_B_ddot, z_W_ddot), finite values
    ├── test_reward.py      vertical terms, velocity scaling, no dead terms
    ├── test_env.py         obs shape, bounds, randomisation, per-profile behaviour
    └── test_curriculum.py  level advancement, road kwargs injection, obs consistency
```

---

## 6. Quickstart

### Installation

```bash
git clone https://github.com/Mohammed-Azab/COptRL.git
cd COptRL

python3 -m venv .venv && source .venv/bin/activate

# CPU-only torch saves ~1.7 GB of disk space
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e src/gym_env
```

### Training

```bash
# PPO with 3-level curriculum (recommended)
python src/train/train.py --algo PPO --road speed_bump --curriculum

# PPO without curriculum — random road every episode, full difficulty from start
python src/train/train.py --algo PPO --road speed_bump

# Resume from checkpoint
python src/train/train.py --algo PPO \
    --resume models/PPO/speed_bump/exp_1/PPO_final.zip

# Override timesteps and parallel envs
python src/train/train.py --algo PPO --road speed_bump \
    --curriculum --timesteps 2000000 --n-envs 4
```

Output layout:
```
models/<ALGO>/<road>/<run_tag>/
    <ALGO>_final.zip          final weights
    vecnormalize.pkl           observation normalisation statistics
    best/best_model.zip        checkpoint with best evaluation return

logs/tensorboard/<run_tag>/   TensorBoard event files
logs/monitor/<run_tag>/       episode-level monitor CSV
```

```bash
tensorboard --logdir logs/tensorboard
```

### Evaluation

```bash
# Deep single-model evaluation — 5 episodes, saves time-series plots
python src/eval/eval.py \
    --algo PPO \
    --model_path models/PPO/speed_bump/exp_1/PPO_final.zip \
    --n-episodes 10 \
    --save-plots

# Agent vs passive and random baselines across all roads
python src/eval/compare.py \
    --algo PPO \
    --model-path models/PPO/speed_bump/exp_1/PPO_final.zip \
    --save-plots
```

`eval.py` plots per episode (7-panel): body accel, wheel accel, speed vs reference, running RMS, action, step reward, reward breakdown.

`compare.py` produces: episode return bars (raw + normalised), RMS accel grouped bars, episode return box plots, representative time-series overlaid per road.

### Hyperparameter Search

```bash
python src/tune/hyperparameter_search.py \
    --algo ppo --trials 50 --timesteps 100000
```

### Tests

```bash
PYTHONPATH=src .venv/bin/pytest tests/ -v
```

---

## 7. Configuration Reference

### `config/reward/reward_params.yaml`

```yaml
weights:
  w_tracking:       0.5
  w_accel:          0.8
  w_jerk:           0.3
  w_action_smooth:  0.2

vertical:
  w_heave:          1.0
  w_wheel:          0.5
  a_B_comfort:      9.81   # m/s²  body accel normaliser
  a_W_comfort:     30.0    # m/s²  wheel accel normaliser
  enable_vel_scaling: true

observations:
  preview_distance:    20.0   # m  lookahead horizon
  h_clip:              0.15   # m  height normalisation clip
  n_peaks:             3      # number of peak slots in observation
  noise_active:        true
  pt1_tau:             0.2    # s  PT1 filter time constant
```

### `config/curriculum/curriculum_params.yaml`

```yaml
thresholds: [200_000, 500_000]   # env steps at which each level unlocks

levels:
  0: {num_bumps_range: [1, 2], bump_height_range: [0.05, 0.10], ...}
  1: {num_bumps_range: [1, 3], bump_height_range: [0.05, 0.15], ...}
  2: {num_bumps_range: [1, 5], bump_height_range: [0.05, 0.25], ...}
```

---

## 8. Key Design Decisions

**Speed planning over active suspension.** The agent has no actuator on the suspension; it only controls longitudinal speed. This makes the problem harder (indirect, predictive control) but eliminates the need for any suspension hardware modification.

**Velocity-scaled reward.** Multiplying the step reward by `v/v_max` prevents the degenerate policy of stopping before every bump (zero excitation = zero penalty). Any policy that slows down too aggressively is penalised by receiving a proportionally smaller reward throughout the deceleration phase.

**Hybrid vertical + longitudinal reward.** Both body acceleration (comfort, ISO 2631) and wheel acceleration (road holding) are penalised, alongside longitudinal jerk and action smoothness. This multi-objective shaping reflects realistic vehicle dynamics requirements.

**Peak-detected preview with noise and filtering.** Raw height grid previews are information-dense but unrealistic (perfect sensor). The peak-encoding with Gaussian noise and PT1 filtering mimics a real-world distance sensor and forces the policy to be robust to imperfect road information.

**Curriculum over full randomisation.** Starting with small, slow bumps and progressively increasing difficulty produces stable early learning. A policy trained on full difficulty from step 1 often fails to learn useful behaviour because early episodes are dominated by large penalties with no gradient signal toward improvement.

**Geometry-safe speed clamping.** `from_random` clamps the sampled vehicle speed to a limit derived from the bump height-to-length ratio, calibrated from two physical test runs. This prevents the training distribution from containing physically unreasonable scenarios (e.g., a 0.25 m bump traversed at 20 m/s).

---

## 9. Dependencies

| Package | Role |
|---------|------|
| Gymnasium ≥ 0.29 | RL environment interface |
| Stable-Baselines3 ≥ 2.3 | PPO / TD3 implementation |
| Numba ≥ 0.59 | JIT-compiled ODE kernels (RK4 at 1 ms) |
| SciPy ≥ 1.11 | `find_peaks` for preview encoding |
| NumPy ≥ 1.24 | Array operations throughout |
| Matplotlib ≥ 3.7 | Evaluation plots |
| PyYAML ≥ 6.0 | Configuration loading |
| Optuna ≥ 3.0 | Hyperparameter search |
| TensorBoard ≥ 2.14 | Training metrics visualisation |
| PyTorch ≥ 2.0 | Neural network backend (CPU-only build sufficient) |

---

*Python 3.10 · Linux / macOS (Parallels)*

# Quarter-Car Speed Planning via Deep Reinforcement Learning

> *Teach a car to slow down before the pothole, not after.*

A quarter-car suspension simulation and training framework where a deep RL agent
learns **longitudinal speed profiles** that minimise ride discomfort (body acceleration,
ISO 2631) while tracking a target velocity. Instead of the classical active-suspension
approach (where the actuator fights the bump in real time), this framing asks:
*can the agent plan its speed so the bump never causes discomfort in the first place?*

Built on Gymnasium · Stable-Baselines3 · Numba.

---

## Problem Framing

The agent observes the suspension state and the upcoming road, then outputs a
normalised acceleration command each 20 ms. The quarter-car ODE integrates the
vertical dynamics passively — the agent's only lever is the longitudinal speed it
arrives at any given point.

```
Observation (up to 14-D) ──► Agent ──► u ∈ [-1, 1]
                                              │
                                    a_cmd = u × 5 m/s²
                                    v_{t+1} = clip(v_t + a_cmd × 0.02, 0, 20)
                                              │
                                    Quarter-car ODE (6-state, RK4)
                                              │
                                    Reward: comfort + tracking + …
```

---

## Repository Layout

```
quarter_car_sim/
│
├── config/                         # All YAML configuration
│   ├── algo/algo_configs.yaml      # SB3 hyperparameters (PPO / SAC / TD3)
│   ├── eval/compare_config.yaml    # Evaluation & comparison settings
│   ├── gym_env/env_params.yaml     # Physics constants, episode length
│   ├── reward/reward_params.yaml   # Reward weights, clips, enable flags
│   ├── road/road_params.yaml       # Road profile parameters
│   └── scenarios/                  # Named experiment scenarios (JSON)
│
├── src/
│   ├── gym_env/QuarterCar_env/     # The Gymnasium environment package
│   │   ├── core/ode_model.py       # 6-state ODE, Numba-jitted RK4, 1 kHz physics
│   │   ├── envs/quarter_car_env.py # Gymnasium Env (obs, step, reset, render)
│   │   ├── reward/reward.py        # Reward terms + reward_bounds()
│   │   ├── config/                 # RewardConfig, env_params, road_params
│   │   └── wrappers/               # ActionRepeat, NormalizeObs, RewardScaler, EpisodeLogger
│   │
│   ├── road/road_generator.py      # ISO 8608, speed bump, sine sweep, flat
│   │
│   ├── train/
│   │   ├── train.py                # Main training entry point
│   │   ├── agent.py                # Algorithm registry (PPO / SAC / TD3)
│   │   ├── environment.py          # make_vec_env / make_eval_vec_env
│   │   ├── monitoring.py           # Callbacks: metrics, checkpoint, sync
│   │   └── seed.py                 # Global seed helper
│   │
│   ├── eval/
│   │   └── compare.py              # Agent vs baseline comparison + plots + JSON
│   │
│   └── tune/
│       └── hyperparameter_search.py  # Ray Tune / Optuna search
│
├── models/                         # Saved model checkpoints (gitignored)
├── logs/                           # TensorBoard + Monitor logs (gitignored)
└── tests/                          # pytest suite
```

---

## The Physics

A **2-DOF quarter-car model** with nonlinear suspension (bumpstop, asymmetric damper):

```
  m_B · z̈_B = −k_S(z_B − z_W) − F_damper(ż_B − ż_W) − F_bumpstop
  m_W · z̈_W =  k_S(z_W − z_B) + F_damper + F_bumpstop − k_T(z_W − ζ) − c_T(ż_W − ζ̇)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| m_B | 465.7 kg | Sprung mass (body) |
| m_W | 50.4 kg | Unsprung mass (wheel) |
| k_S | 27 922 N/m | Suspension spring stiffness |
| k_T | 262 200 N/m | Tyre radial stiffness |
| c_T | 500 N·s/m | Tyre damping |

Integration: **RK4 with 16 sub-steps** per 20 ms control step (≈ 0.5 ms physics step).
All integration is Numba-jitted — a full 250-step episode takes < 2 ms.

### 6-State Vector

```
x[0] = ζ − z_W    tyre deflection         [m]
x[1] = ż_W        wheel vertical velocity  [m/s]
x[2] = z_W − z_B  suspension travel        [m]
x[3] = ż_B        body vertical velocity   [m/s]
x[4] = v          longitudinal speed       [m/s]   ← driven by agent action
x[5] = z_B        body displacement        [m]
```

---

## Observation Space

Up to **14-dimensional** float32 vector (exact size depends on `obs_enable_*` flags):

| Index | Signal | Description |
|-------|--------|-------------|
| 0 | z_B | body displacement [m] |
| 1 | ż_B | body vertical velocity [m/s] |
| 2 | z_W | wheel displacement [m] |
| 3 | ż_W | wheel vertical velocity [m/s] |
| 4 | ζ | road height [m] |
| 5 | ζ̇ | road vertical velocity [m/s] |
| 6 | z_W − z_B | suspension travel [m] |
| 7 | ζ − z_W | tyre deflection [m] |
| 8 | v / v_max | normalised current speed |
| 9 | (v_target − v) / v_max | normalised speed error |
| 10 | filtered_a / a_comfort | normalised smoothed accel (if enabled) |
| 11 | filtered_jerk / j_max | normalised smoothed jerk (if enabled) |
| 12 | prev_action | previous command (if enabled) |
| 13 | curvature / clip | road curvature (if enabled) |

**Action space:** Box([-1], [1]) — normalised acceleration command.

---

## Reward Function

The reward is designed so that:
- **Episode max = +300** (achievable by a perfect agent)
- **Practical range ≈ [-500, +300]** (what real agents experience)
- **Theoretical minimum = -1350** (all terms at clip boundaries simultaneously — never reached in practice due to IIR filtering)

```
R = w_comfort_bonus · r_comfort_bonus    ← [  0,  +1]  w=0.8
  + w_tracking      · r_tracking        ← [ -1,   0]  w=1.0
  + w_accel         · r_accel           ← [ -4,   0]  w=0.5
  + w_jerk          · r_jerk            ← [ -4,   0]  w=0.3
  + w_action_smooth · r_action_smooth   ← [ -4,   0]  w=0.2
  + w_curve         · r_curve           ← disabled     w=0.0
```

### Term Details

| Term | Formula | Range | Role |
|------|---------|-------|------|
| `r_comfort_bonus` | `max(0, 1 − (a_filt/a_comfort)²)` | [0, +1] | positive signal for smooth riding |
| `r_tracking` | `−((v − v_target)/v_max)²` | [-1, 0] | stay near reference speed |
| `r_accel` | `−(a_filt/a_comfort)²` clipped at 2×a_comfort | [-4, 0] | ISO 2631 comfort (longitudinal) |
| `r_jerk` | `−(j_filt/j_max)²` clipped at 2×j_max | [-4, 0] | smooth acceleration changes |
| `r_action_smooth` | `−(u_t − u_{t-1})²` | [-4, 0] | continuous control commands |

**Terminal signal:** `+100` if episode RMS body accel < 10 m/s², else `-100`.

**Clips:** `reward_accel_clip = 6 m/s²` (= 2×a_comfort) and `reward_jerk_clip = 20 m/s³`
(= 2×j_max) apply only to reward computation. Observation clips remain at 15 m/s² and 50 m/s³
for numerical safety — no trained-model breakage when changing reward clips.

---

## Road Profiles

| Profile | Description | Primary use |
|---------|-------------|-------------|
| `iso_8608_class_c` | PSD-synthesised random roughness (class C) | Training; general robustness |
| `speed_bump` | Versine bump, configurable height and length | Impulse/comfort evaluation |
| `sine_sweep` | 0.5 → 20 Hz chirp over episode | Frequency-response analysis |
| `flat` | ζ = 0 | Baseline / speed-control isolation |

---

## Quickstart

### Install

```bash
git clone <repo>
cd quarter_car_sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/gym_env
```

### Train

```bash
# PPO on speed-bump road (default 1 M steps)
python src/train/train.py --algo PPO --road speed_bump

# SAC on ISO 8608 Class C, custom timesteps and seed
python src/train/train.py --algo SAC --road iso_8608_class_c --timesteps 500000 --seed 42

# TD3, resume from checkpoint
python src/train/train.py --algo TD3 --resume models/TD3/my_run/checkpoints/ckpt_50000_steps.zip

# Disable VecNormalize (off-policy only)
python src/train/train.py --algo SAC --no-normalize
```

Training writes to:
```
models/<ALGO>/<run_tag>/
    <algo>_final.zip          # final model
    vecnormalize.pkl           # observation normalisation stats
    best/best_model.zip        # best checkpoint (by eval reward)
    checkpoints/               # periodic checkpoints + vecnormalize

logs/tensorboard/<run_tag>/   # TensorBoard event files
logs/monitor/<run_tag>/       # Episode monitor CSV
```

Monitor training:
```bash
tensorboard --logdir logs/tensorboard
```

### Evaluate and Compare

```bash
# Compare trained agent vs passive and random baselines on all roads
python src/eval/compare.py \
    --algo PPO \
    --model-path models/PPO/my_run/PPO_final.zip \
    --save-graphs

# Single road, more episodes
python src/eval/compare.py \
    --algo PPO \
    --model-path models/PPO/my_run/PPO_final.zip \
    --road speed_bump --n-episodes 30 --save-graphs

# Render the simulation live
python src/eval/compare.py \
    --algo PPO \
    --model-path models/PPO/my_run/PPO_final.zip \
    --render

# Use a custom config file
python src/eval/compare.py --config config/eval/compare_config.yaml
```

`compare.py` produces:
- **Console table** — mean ± std for return, RMS accel, speed RMSE, comfort score
- **`returns.png`** — raw episode return + normalised % of optimal, with theoretical bounds as reference lines
- **`rms_accel.png`** — grouped bar chart with ISO 2631 comfort/discomfort thresholds
- **`return_distributions.png`** — box plots per road, per agent
- **`timeseries_<road>.png`** — body accel / speed / action / step-reward overlaid for all agents
- **`compare_<ALGO>_<timestamp>.json`** — full metrics export

### Hyperparameter Search

```bash
python src/tune/hyperparameter_search.py --algo PPO --trials 50 --timesteps 100000
```

---

## Configuration

All behaviour is controlled through YAML files in `config/`. CLI flags override YAML defaults.

### `config/algo/algo_configs.yaml`

SB3 constructor kwargs for PPO, SAC, TD3 plus training meta-settings (timesteps, n_envs,
eval frequency, checkpoint frequency, normalisation flags).

### `config/reward/reward_params.yaml`

```yaml
weights:
  w_comfort_bonus:  0.8   # per-step positive bonus (drives episode_max to +300)
  w_tracking:       0.8
  w_accel:          0.5
  w_jerk:           0.3
  w_action_smooth:  0.2

comfort:
  a_comfort:          3.0   # m/s²  ISO 2631 comfort threshold
  reward_accel_clip:  6.0   # m/s²  reward clip = 2 × a_comfort
  accel_clip:        15.0   # m/s²  observation clip (separate, unchanged)

terminal:
  terminal_bonus:   100.0
  terminal_penalty: -100.0
  a_limit:          10.0    # m/s²  RMS threshold for terminal bonus
```

### `config/eval/compare_config.yaml`

Controls `compare.py`: which roads to evaluate, how many episodes, which baselines,
whether to render or save graphs.

---

## Algorithms

| Algorithm | Policy | Notes |
|-----------|--------|-------|
| **PPO** | MLP [256, 256] (pi + vf) | On-policy; reward normalisation on |
| **SAC** | MLP [256, 256] | Off-policy; no reward normalisation |
| **TD3** | MLP [400, 300] | Off-policy; more stable, slower |

All algorithms use `VecNormalize` for observation normalisation (running mean/std).
At evaluation time the stats are frozen (`training=False`) and loaded from the saved
`.pkl` file alongside the model checkpoint.

---

## Baselines in `compare.py`

| Baseline | Policy | What it tells you |
|----------|--------|-------------------|
| `passive` | `u = 0.0` always | Constant speed cruise control — RL lower bound |
| `random` | `u ~ Uniform[-1, 1]` | Absolute lower bound; sanity check |

---

## Key Design Decisions

**Speed planning, not active suspension.** The agent cannot directly push the car body —
it can only choose how fast to go. This makes the problem harder (indirect control) but
more realistic for road vehicles without active suspension hardware.

**Separate reward and observation clips.** `reward_accel_clip` and `reward_jerk_clip`
bound the worst-case reward penalty without changing the observation space. This decouples
"what the agent sees" from "how bad the penalty is", and means changing reward clips
never invalidates trained models.

**IIR-filtered signals in reward.** Raw finite-difference acceleration is noisy. The
reward uses exponentially-smoothed signals (α = 0.8) which are also fed back as
observations, giving the agent and the reward the same filtered view of dynamics.

**Terminal bonus.** A sparse +100 / -100 signal at episode end based on episode RMS
body acceleration adds a longer-horizon objective on top of the dense per-step reward,
encouraging the agent to maintain comfort throughout not just locally.

---

## References

Full citations in [`refs.txt`](refs.txt). Key works:

- ISO 8608:2016 — Road surface profiles, classification
- ISO 2631-1:1997 — Mechanical vibration and shock, human exposure
- Raffin et al. — Stable-Baselines3 *(JMLR 2021)*
- Nhu et al. — Physics-Guided RL for Vehicle Suspension *(ICMLA 2023)*

---

*Python 3.10+ · Stable-Baselines3 · Gymnasium · Numba · Matplotlib*

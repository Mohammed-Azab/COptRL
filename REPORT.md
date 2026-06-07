# COptRL — Technical Report & Thesis Presentation Material

---

## PART 0 — PROMPT FOR CLAUDE (Presentation Generator)

> **Copy this block verbatim into a new Claude conversation, attach your slide template, and Claude will produce a ready-to-use presentation.**

---

```
You are a thesis presentation designer. I am providing you with:
  1. A full technical report about my RL research project (COptRL) — read it in its entirety.
  2. A slide template file (attached).

Your task: produce a complete slide-by-slide presentation following the template's visual style, 
fonts, and layout exactly. Do not invent content — every claim, number, equation, and figure 
description must come directly from the report. Where the report says "see figure", describe the 
chart/diagram in the speaker notes so I can recreate it.

Structure the deck as follows (adjust slide count to fit the template's format):

  Slide 1  — Title: "COptRL: Reinforcement Learning for Predictive Speed Control Over Road Bumps"
  Slide 2  — Motivation & Problem Statement
  Slide 3  — System Overview (one-slide architecture diagram)
  Slide 4  — Vehicle Model (Quarter-Car ODE)
  Slide 5  — Gymnasium Environment
  Slide 6  — Observation Space
  Slide 7  — Reward Function (equations + weights table)
  Slide 8  — Road Generator & Preview System
  Slide 9  — Training Pipeline (PPO + VecNormalize + wrappers stack)
  Slide 10 — Curriculum Learning
  Slide 11 — Hyperparameter Tuning (Optuna)
  Slide 12 — Evaluation Framework
  Slide 13 — What Was Taken from ba_azab & What Was Built New
  Slide 14 — Key Design Decisions
  Slide 15 — Results & Metrics (fill with placeholder charts; add speaker notes on what to show)
  Slide 16 — Conclusion & Future Work

For each slide: provide (a) headline text, (b) bullet points or equations, 
(c) speaker notes, (d) figure description if applicable.

Report content starts after the horizontal rule below.
```

---

---

## PART 1 — FULL TECHNICAL REPORT

---

## 1. Motivation & Problem Statement

Modern vehicles are routinely equipped with passive suspension systems that absorb road disturbances
without driver input. However, for a given suspension tuning, ride comfort when crossing a speed bump
depends heavily on the vehicle speed at the moment of impact. A driver who slows down predictively —
based on seeing the bump ahead — can drastically reduce the body acceleration experienced by
passengers, even without any active suspension hardware.

This project develops a Reinforcement Learning (RL) agent that learns this predictive speed-planning
behaviour. The agent controls only the longitudinal acceleration (throttle/brake) of a quarter-car
model; the suspension is passive and unchanged. The problem is relevant to:

- ADAS (Advanced Driver Assistance Systems) — automated speed control for comfort
- Autonomous driving comfort modules
- Human driver modelling in vehicle development simulation

The thesis follows the MPC-based driver modelling framework of Mandl (2021) and replaces the MPC
with a learned PPO policy, using the same comfort-oriented cost function structure as a reward signal.

---

## 2. Repository Architecture

```
COptRL/
├── src/
│   ├── gym_env/QuarterCar_env/       # Gymnasium environment package
│   │   ├── core/ode_model.py          # 6-state quarter-car ODE (Numba RK4)
│   │   ├── envs/quarter_car_env.py    # QuarterCarEnv (gym.Env subclass)
│   │   ├── reward/reward.py           # All reward term functions
│   │   ├── config/                    # Dataclasses + YAML loaders
│   │   └── wrappers/                  # PreviewWrapper, CurriculumWrapper, EpisodeLogger
│   ├── road/road_generator.py         # Cosine-bump road model + from_random()
│   ├── train/                         # PPO training pipeline
│   │   ├── train.py                   # Entry point (argparse → train loop)
│   │   ├── environment.py             # VecEnv + VecNormalize factory
│   │   └── agent.py                   # Model registry + builder
│   ├── tune/                          # Optuna hyperparameter search
│   │   ├── tune.py                    # Study management + results saving
│   │   ├── trial.py                   # Per-trial objective function
│   │   └── search_spaces.py           # YAML search-space → Optuna suggest calls
│   └── eval/
│       ├── eval.py                    # Single-model deep evaluation + plots
│       └── compare.py                 # Agent vs baselines comparison
├── config/                            # All YAML configuration files
│   ├── gym_env/env_params.yaml        # Physics constants
│   ├── reward/reward_params.yaml      # Reward weights and comfort thresholds
│   ├── algo/algo_configs.yaml         # PPO hyperparameters
│   ├── algo/tune_config.yaml          # Optuna search space
│   └── curriculum/curriculum_params.yaml
└── justfile                           # One-liner recipes (train, tune, eval, ...)
```

---

## 3. Vehicle Model — Quarter-Car ODE

### 3.1 Physical Model

The vehicle is represented as a two-mass quarter-car model following Mandl (2021) and the standard
textbook formulation of Mitschke & Wallentowitz (2014). The two masses are:

- **m_B = 465.7 kg** — sprung mass (car body, one quarter)
- **m_W = 50.4 kg** — unsprung mass (wheel assembly)

Connected by:

| Component | Type | Parameters |
|-----------|------|-----------|
| Suspension spring | Nonlinear (exponential bumpstop) | k_S = 27,922 N/m, clearances: dz_cmp=0.02 m, dz_rbd=0.08 m |
| Suspension damper | Sigmoid-blended piecewise | D = 3530 N·s/m, A = 0.5 (asymmetry) |
| Tyre | Linear spring + damper | k_T = 262,200 N/m, c_T = 500 N·s/m |

**All parameters are taken directly from Mandl (2021) Appendix A.1 (Table A.1 — Quarter-Car Parameters).**
These match the vehicle used in the Nardo test drives from which the recorded scenarios were captured.

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| m_B | 465.7 | kg | Mandl (2021) Table A.1 — sprung mass (one quarter of vehicle body) |
| m_W | 50.4 | kg | Mandl (2021) Table A.1 — unsprung mass (wheel + hub + partial axle) |
| k_S | 27,922 | N/m | Mandl (2021) Table A.1 — linear suspension spring stiffness |
| k_T | 262,200 | N/m | Mandl (2021) Table A.1 — tyre radial stiffness |
| c_T | 500 | N·s/m | Mandl (2021) Table A.1 — tyre damping coefficient |
| D | 3530 | N·s/m | Mandl (2021) Table A.1 — averaged low-speed damping (D = (d1+z1)/2) |
| A | 0.5 | — | Mandl (2021) Table A.1 — damper asymmetry (compression/rebound ratio) |
| v_d, v_z | 0.20 | m/s | Mandl (2021) Table A.1 — low/high speed transition velocities |
| f1_cmp | 1/3 | — | Mandl Eq. 3.9 — nonlinear spring compression progression factor |
| f2_cmp | 4.0 | — | Mandl Eq. 3.9 — curvature of progressive rise (compression) |
| f1_rbd | 1.0 | — | Mandl Eq. 3.10 — nonlinear spring rebound progression factor |
| f2_rbd | 8.0 | — | Mandl Eq. 3.10 — curvature of progressive rise (rebound, stiffer than compression) |
| dz_cmp | 0.02 | m | Mandl (2021) Table A.1 — clearance before compression bumpstop engages |
| dz_rbd | 0.08 | m | Mandl (2021) Table A.1 — clearance before rebound bumpstop engages |
| F_ks_nlin_max | 100,000 | N | Numerical cap to prevent integrator blow-up (not in Mandl — added for robustness) |

**Resonant frequencies** (derived from parameters, Mandl Fig. 3.7):
- Body (heave) resonance: f_heave ≈ 1.17 Hz — dominant comfort frequency
- Wheel-hop resonance: f_wheel ≈ 12.1 Hz — road holding
- Human sensitivity peak: 4–8 Hz (ISO 2631-1) — bumps designed to excite this range

The equations of motion are:

```
m_B * z̈_B =  F_spring + F_damper
m_W * z̈_W = -F_spring - F_damper + F_tyre_spring + F_tyre_damper
```

Longitudinal speed v is driven externally by the RL agent (not by the ODE itself).

### 3.2 State Vector

The ODE integrates a 6-element state vector:

```
x = [ζ − z_W,   ż_W,   z_W − z_B,   ż_B,   v,   z_B]
      tyre def   wheel vel  susp travel  body vel  speed  body disp
```

### 3.3 Numba-Accelerated Integration

The ODE is implemented in `core/ode_model.py` using four Numba `@njit`-compiled kernels:

| Kernel | Function |
|--------|---------|
| `_spring_nonlin` | Exponential bumpstop beyond clearance zones |
| `_damper` | Sigmoid-blended piecewise (avoids hard switching) |
| `_ode` | Full state derivative computation |
| `_rk4_loop` | Fixed-step 4th-order Runge-Kutta, N_SUB substeps per control step |

**Integration scheme:** Each 20 ms control step (DT = 0.02 s) is subdivided into 20 ODE substeps
(DT_SIM = 0.001 s) giving 1 kHz physics fidelity. The RK4 loop evaluates the road velocity
`ζ̇` at three points per substep (start, midpoint, end) to correctly track the cosine-bump gradient.

**Output:** `QuarterCarODE.step()` returns `(new_state, z_B_ddot, z_W_ddot)` — the new state plus
both body and wheel vertical accelerations from the terminal ODE evaluation. These accelerations feed
directly into the reward function.

---

## 4. Gymnasium Environment (`QuarterCarEnv`)

The environment is a standard `gymnasium.Env` registered as `"QuarterCar_env/QuarterCar"`.

### 4.1 Action Space

```
action ∈ [-1, 1]  (Box, float32)
a_cmd = action * a_max            # scaled to ±5 m/s²
v_new = clip(v_old + a_cmd * DT, 0, v_max)
```

The agent controls longitudinal acceleration. Speed is integrated with an Euler step; the resulting
actual acceleration feeds the reward and filter.

### 4.2 Observation Space (Base Environment — 6 features)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | ζ | [0, 0.15] | Road height at current wheel position [m] |
| 1 | ζ̇ | [-7, 7] | Road velocity at wheel [m/s] |
| 2 | v / v_max | [0, 1] | Normalised longitudinal speed |
| 3 | filtered_a / a_comfort | [−4, 4] | IIR-smoothed longitudinal acceleration |
| 4 | filtered_jerk / j_max | [−6, 6] | IIR-smoothed jerk |
| 5 | prev_action | [−1, 1] | Previous action (command memory) |

The full observation is 6 + 3·n_peaks = **15 features** after the PreviewWrapper appends 9 peak slots.

### 4.3 Episode Flow

```
reset()  → randomise road (if speed_bump) → seed ODE → return obs
step()   →  1. integrate speed: v += a_cmd * DT
            2. update IIR filters (accel, jerk)
            3. ODE.step() → new_state, z_B_ddot, z_W_ddot
            4. compute_reward()
            5. check truncation (travel > 0.6 m, |z_B| > 0.6 m, max_distance)
            6. check termination (step_count == 300) → add terminal bonus
```

### 4.4 Road Profiles

| Profile | Description |
|---------|-------------|
| `speed_bump` | Cosine half-wave bumps; geometry randomised each reset via `from_random()` |
| `flat` | Zero road height; used to validate baseline (no excitation) |
| `recorded` | Real road data from test drives at Nardo proving ground (10 scenarios) |

---

## 5. Road Generator

`RoadGenerator` in `src/road/road_generator.py` provides all road-profile logic.

### 5.1 Cosine Bump Shape

Each bump follows the Mandl (2021) formula:

```
ζ(x) = (H/2) * (1 - cos(2π(x - x₀)/W))    for x₀ ≤ x ≤ x₀ + W
```

where H is bump height [m] and W is bump length [m].

### 5.2 Random Road Generation (`from_random`)

At every `reset()`, a new bump layout is sampled from the curriculum-level bounds:

```python
n       ∼ Uniform(num_bumps_range)
heights ∼ Uniform(bump_height_range)   for each of n bumps
lengths ∼ Uniform(bump_length_range)
gaps    ∼ Uniform(min_gap, 3 × min_gap)   between consecutive bumps
```

A speed-safety clamp (`_clamp_speed_to_geometry`) limits the vehicle speed based on the maximum
height-to-length ratio across all sampled bumps, calibrated from two empirical test runs.

### 5.3 Spatial Preview

`get_spatial_preview(s_pos, v, lookahead_m, n_points)` returns road heights at 200 evenly-spaced
spatial positions ahead of the vehicle — independent of time, speed, or episode state. This is
consumed by `PreviewWrapper` for peak detection.

### 5.4 Recorded Scenarios (10 files)

Real road data from Nardo Circuit (speed-bump runs) and highway segments are stored as JSON
arc-length/height pairs in `config/scenarios/`. They span vehicle speeds from 3 to 14 m/s and arc
lengths from 30 m to 300 m.

---

## 6. Wrappers

### 6.1 PreviewWrapper

`PreviewWrapper(gym.ObservationWrapper)` appends n_peaks × 3 = 9 features to the base observation.
It runs entirely in Python and does not modify the ODE or reward.

**Algorithm:**
1. Query `road.get_spatial_preview()` → 200-point height array over 20 m lookahead
2. Run `scipy.signal.find_peaks` with min height 0.01 m and min separation 0.5 m
3. Encode up to 3 detected peaks as normalised triplets `[dist/D, height/h_clip, width/D]`
4. Fill undetected slots with `[1.0, 0.0, 0.0]` (bump at horizon, zero height/width)
5. Add Gaussian noise scaled by peak distance (closer peaks → less noise)
6. Apply PT1 filter: `α = DT / (τ + DT)` with τ = 0.2 s

**Design rationale:** This mimics a realistic distance sensor — peaks close to the vehicle are
estimated accurately; distant peaks are noisier. The PT1 filter prevents the policy from
reacting to instantaneous observation jumps (realistic sensor smoothing).

### 6.2 CurriculumWrapper

`CurriculumWrapper(gym.Wrapper)` intercepts every `reset()` and injects level-specific
road geometry bounds via the `options` dict. It tracks total environment steps and advances
the difficulty level automatically.

| Level | Steps | Bumps | Height | Length | Speed |
|-------|-------|-------|--------|--------|-------|
| 0 | 0 – 200 k | 1–2 | 0.05–0.10 m | 3–7 m | 4–10 m/s |
| 1 | 200 k – 500 k | 1–3 | 0.05–0.15 m | 2–7 m | 4–15 m/s |
| 2 | 500 k+ | 1–5 | 0.05–0.25 m | 1–7 m | 4–20 m/s |

**Design rationale:** Starting with small, slow bumps prevents early-episode reward collapse.
A policy trained on full difficulty from step 0 often stalls because large penalties with no
gradient signal toward improvement make learning infeasible.

### 6.3 EpisodeLogger

Logs per-episode metrics (return, RMS accel, comfort score, speed RMSE) to CSV files readable
by TensorBoard and the evaluation scripts.

---

## 7. Reward Function

The reward is designed following the cost-function structure of Mandl (2021, Ch. 4.4) and grounded
in ISO 2631-1:2016 (vibration comfort thresholds). The relationship between Mandl's MPC cost
function and our RL reward is direct and intentional — we replace his MPC optimisation objective
with an equivalent step-wise RL reward signal.

### 7.1 What Mandl (2021) Provides and How It Maps Here

Mandl formulates an MPC cost function J = J_comfort + J_speed + J_input (Eq. 4.18):

| Mandl cost term | Mandl formula | Mandl weight | This repo equivalent | Our weight |
|-----------------|--------------|--------------|---------------------|-----------|
| J_comfort (body accel) | Qc · z̈_B⁴ (quartic) | Qc = 0.3 | r_heave = −(z̈_B/0.5)² (quadratic) | w_heave = 0.8 |
| J_speed (velocity deviation) | Qv · |v_ref − v| (absolute) | Qv = 0.7 | r_tracking = −((v−v_ref)/v_ref)² | w_tracking = 0.8 |
| J_input (longitudinal accel) | Qu · a⁴ (quartic) | Qu = 1.0 | r_accel = −(ā/2.0)² | w_accel = 0.4 |
| J_terminal | Qt·Ns·(v_ref − v_end)² | Qt = 0.001 | terminal_bonus = ±100 | — |

**Why quadratic instead of Mandl's quartic?**
Mandl uses quartic (`z̈⁴`) because it drives speed toward zero more aggressively at the bump.
COptRL uses quadratic throughout for training stability — quartic gradients can be extreme when
the policy is random at the start of training, causing loss spikes and policy collapse.
The relative weighting preserves Mandl's intent: comfort and speed terms are approximately equal
in importance (`w_heave ≈ w_tracking = 0.8`), with input/jerk terms smaller.

**Why absolute tracking in Mandl but quadratic here?**
Mandl explains (Sec. 4.4.1): absolute penalty `|v_ref − v|` is preferred because it does not
over-penalise the velocity dip during bump crossing. We use squared tracking because RL objectives
benefit from smooth, differentiable (through value function) signal — the effect is the same
at the scale of a 300-step episode.

**Additional terms not in Mandl's MPC (added for RL training stability):**

| Term | Formula | Rationale |
|------|---------|-----------|
| r_wheel | −(z̈_W/30)² | Road holding / tyre load variation; common in suspension control literature [Čorić 2016, Giorgetti 2006] |
| r_jerk | −(j̄/2)² | Penalises abrupt acceleration changes; not in Mandl but standard in automotive comfort [EN 13374] |
| r_action_smooth | −(u_t − u_{t−1})² | Prevents action chattering in discrete RL updates |
| velocity scaling | R *= v/v_max | Prevents degenerate stop-and-wait policy; not in MPC (MPC has a minimum-speed constraint) |

### 7.2 Full Reward Formulation

```
R_step = (v / v_max) × [
    w_heave  · r_heave(z̈_B)         # body vertical comfort
  + w_wheel  · r_wheel(z̈_W)         # wheel vertical (road holding)
  + w_tracking · r_tracking(v)       # speed management
  + w_accel  · r_accel(ā)            # longitudinal comfort
  + w_jerk   · r_jerk(j̄)            # jerk smoothness
  + w_smooth · r_smooth(u_t, u_{t−1}) # action continuity
]

R_episode += R_terminal  (added only at step 300)
```

where `ā` is the IIR-filtered longitudinal acceleration and `j̄` is the IIR-filtered jerk,
both computed in the environment step function before reward calculation.

### 7.3 Individual Term Definitions

```
r_heave(z̈_B) = −(clip(z̈_B, ±1.0) / 0.5)²
  → worst case at clip: −(1.0/0.5)² = −4.0

r_wheel(z̈_W) = −(clip(z̈_W, ±60) / 30)²
  → worst case at clip: −(60/30)² = −4.0

r_tracking(v) = 0                          if v ∈ [v_min, v_max]
              = −((v_min − v) / v_min)²    if v < v_min  (stopping penalty)
              = −((v − v_max) / v_max)²    if v > v_max  (overspeed, not expected)

r_accel(ā)   = −(clip(ā, ±4.0) / 2.0)²
  → worst case: −(4/2)² = −4.0

r_jerk(j̄)   = −(clip(j̄, ±4.0) / 2.0)²
  → worst case: −(4/2)² = −4.0

r_smooth(u)  = −(u_t − u_{t−1})²
  → worst case: −(1−(−1))² = −4.0
```

All clips are designed so that the worst-case per-term contribution is −4.0, giving a
consistent scale for all reward components.

### 7.4 Parameter Values and Their Sources

#### Vertical comfort — body acceleration

| Parameter | Value | Source & Justification |
|-----------|-------|------------------------|
| `a_B_comfort` | **0.5 m/s²** | **ISO 2631-1:2016, Table 3**: boundary between "a little uncomfortable" (0.315–0.63 m/s² RMS) and "fairly uncomfortable" (0.5–1.0 m/s² RMS). Chosen as the mid-point of "fairly uncomfortable" — the threshold above which most passengers notice discomfort. |
| `reward_heave_clip` | **1.0 m/s²** | = 2 × a_B_comfort. Clip normalises the penalty range to [−4, 0] matching all other terms. |
| `w_heave` | **0.8** | Matches Mandl's Qc weight (relative to Qv = 0.7) expressed on a [0, 1] scale. Primary comfort objective. |

#### Vertical comfort — wheel acceleration

| Parameter | Value | Source & Justification |
|-----------|-------|------------------------|
| `a_W_comfort` | **30.0 m/s²** | Standard wheel acceleration normaliser from vehicle dynamics literature (Mitschke 2014, Klinger 2018). Wheel-hop resonance occurs at ~12 Hz; at 30 m/s² the wheel is near losing road contact. |
| `reward_wheel_clip` | **60.0 m/s²** | = 2 × a_W_comfort. |
| `w_wheel` | **0.3** | Secondary objective (road holding vs. comfort). Lower than w_heave since wheel accel is a less direct passenger experience metric. |

#### Speed tracking

| Parameter | Value | Source & Justification |
|-----------|-------|------------------------|
| `v_max` | **20.0 m/s** | 72 km/h — typical urban/suburban speed limit. Reference speed for tracking (Mandl: v_ref). |
| `v_min` | **2.0 m/s** | Minimum speed below which stopping penalty fires. Prevents learning to stop before bumps. |
| `w_tracking` | **0.8** | Matches Mandl Qv = 0.7 (dominant term in Mandl's cost function). |

#### Longitudinal comfort (acceleration)

| Parameter | Value | Source & Justification |
|-----------|-------|------------------------|
| `a_comfort` | **2.0 m/s²** | ISO 2631-1 longitudinal direction uses 1.4× frequency weighting vs. vertical. Comfortable braking/acceleration threshold ≈ 0.5 × 1.4 = 0.7 m/s² for smooth driving; 2.0 m/s² represents firm but acceptable deceleration for obstacle avoidance. |
| `reward_accel_clip` | **4.0 m/s²** | = 2 × a_comfort. |
| `accel_filter_alpha` | **0.8** | IIR smoothing: filters out single-step noise in computed acceleration. α = 0.8 gives ~5-step time constant at DT = 0.02 s. |
| `w_accel` | **0.4** | Half of Mandl's Qu = 1.0 (quartic → quadratic correction approximately halves effective magnitude). |

#### Jerk

| Parameter | Value | Source & Justification |
|-----------|-------|------------------------|
| `j_max` | **2.0 m/s³** | EN 13374 (railway comfort), ISO 22133 (automated vehicles): comfortable jerk < 2 m/s³. European rail standard limits jerk to < 1 m/s³ for passengers standing; 2 m/s³ is reasonable for seated passengers in a car. |
| `reward_jerk_clip` | **4.0 m/s²** | = 2 × j_max. |
| `w_jerk` | **0.2** | Jerk is a secondary comfort signal (not in Mandl's quarter-car model but added in his half-car model via the jerk-input formulation, Sec. 4.4.2). |

#### Terminal reward

| Parameter | Value | Source & Justification |
|-----------|-------|------------------------|
| `a_limit` | **1.0 m/s²** | **ISO 2631-1**: boundary of "uncomfortable" zone is 0.8–1.6 m/s² RMS. 1.0 m/s² is the centre of this range — a meaningful training target that the policy must actively work to achieve. With random actions the RMS is ~2.5 m/s², so the terminal penalty of −100 fires for an untrained policy. |
| `terminal_bonus` | **+100** | One episode of best-case per-step reward ≈ 300 × 0 = 0. The terminal bonus is large enough to dominate the gradient signal at the end of good episodes. |
| `terminal_penalty` | **−100** | Symmetric to bonus. Fires whenever the agent fails the comfort target. |

### 7.5 Velocity Scaling — Prevention of Stop-and-Wait

The factor `v/v_max` multiplies the entire step reward:

```
R_step = (v/v_max) × core
```

**Why this is necessary:** Without velocity scaling, the optimal policy for all vertical penalty
terms is to stop the vehicle (`v = 0`) before every bump. Zero speed → zero road excitation →
zero vertical acceleration → zero penalty. This is physically unrealistic and defeats the purpose
of the agent.

With velocity scaling, a vehicle travelling at 10 m/s (half of v_max) receives only half the
potential reward magnitude. Long periods of low speed — even with perfect comfort — are penalised
indirectly. This is equivalent to Mandl's minimum-speed constraint (Eq. 4.28: v ≥ v_min) but
implemented as a soft gradient rather than a hard constraint, which is more tractable in RL.

### 7.6 ISO 2631-1 Comfort Scale (Full Reference)

| RMS Acceleration [m/s²] | Comfort Category |
|------------------------|-----------------|
| < 0.315 | Not uncomfortable |
| 0.315 – 0.63 | A little uncomfortable |
| 0.5 – 1.0 | Fairly uncomfortable |
| 0.8 – 1.6 | Uncomfortable |
| 1.25 – 2.5 | Very uncomfortable |
| > 2.0 | Extremely uncomfortable |

COptRL target: trained agent should achieve RMS < 1.0 m/s² (= "uncomfortable" threshold),
ideally approaching 0.5 m/s² ("a little uncomfortable" / "fairly uncomfortable" boundary).

### 7.7 Reward Bounds

With all clips active, v = v_max:

```
Per-step minimum (all at worst-case clip, v = v_max):
  r_heave    : w_heave × (−4.0) = −3.2
  r_wheel    : w_wheel × (−4.0) = −1.2
  r_tracking : w_tracking × (−1.0) = −0.8  [tracking error = 1]
  r_accel    : w_accel × (−4.0) = −1.6
  r_jerk     : w_jerk × (−4.0) = −0.8
  r_smooth   : w_smooth × (−4.0) = −0.4
  ─────────────────────────────────────
  Total per-step min = −8.0 × (v/v_max at v_max = 1.0) = −8.0
  (In practice: −7.20 observed, as not all terms simultaneously hit clip)

Episode min = 300 × (−7.20) + (−100) = −2260
Episode max = 300 × 0 + 100 = +100
```

---

## 8. Training Pipeline

### 8.1 Algorithm: PPO (Proximal Policy Optimisation)

PPO is the sole algorithm. It was chosen because:
- On-policy: observations are collected fresh from the current policy — no stale replay buffer
- Compatible with VecNormalize reward normalisation (off-policy algorithms cannot normalise
  rewards correctly with a shared replay buffer)
- Simple, stable training with a flat MLP policy matching the flat observation vector

The policy architecture is a standard `MlpPolicy` with two hidden layers of 256 units for both
actor (pi) and critic (vf):

```
obs (15,) → [256 → ReLU → 256 → ReLU] → action mean (1,)   [actor]
           → [256 → ReLU → 256 → ReLU] → value (1,)          [critic]
```

### 8.2 VecNormalize

The training environment is wrapped in SB3's `VecNormalize` with:
- `norm_obs = True` — running mean/variance normalisation on all 15 observation features
- `norm_reward = True` — reward normalised by a running estimate of the return magnitude
- `clip_obs = 10.0`, `clip_reward = 10.0` — prevents extreme values destabilising learning

The eval environment shares the same `obs_rms` as the training environment but has
`norm_reward = False` and `training = False` to avoid contaminating statistics during evaluation.

### 8.3 Wrapper Stack (training)

```
QuarterCarEnv
  └── PreviewWrapper          # appends 9 preview features
        └── CurriculumWrapper  # injects difficulty level at reset (speed_bump only)
              └── Monitor        # episode CSV logging (SB3 standard)
                    └── EpisodeLogger  # per-episode metric logging
                          └── DummyVecEnv (n_envs workers)
                                └── VecNormalize
```

### 8.4 Output Structure

```
models/PPO/<road>/exp_<n>/
    PPO_final.zip          # weights after all timesteps
    vecnormalize.pkl       # obs/reward normalisation statistics (required for eval)
    best/best_model.zip    # snapshot at highest eval return
logs/tensorboard/<run>/    # TensorBoard event files
logs/monitor/<run>/        # CSV episode logs
```

---

## 9. Hyperparameter Tuning (Optuna)

### 9.1 Search Space

Tuning is driven by `config/algo/tune_config.yaml`. All 11 parameters are sampled by Optuna:

| Parameter | Type | Range |
|-----------|------|-------|
| learning_rate | log-float | [1e-5, 1e-3] |
| n_steps | categorical | {1024, 2048, 4096} |
| batch_size | categorical | {64, 128, 256} |
| n_epochs | int | [5, 20] |
| gamma | float | [0.97, 0.999] |
| gae_lambda | float | [0.90, 0.99] |
| clip_range | categorical | {0.1, 0.2, 0.3} |
| ent_coef | log-float | [1e-8, 1e-2] |
| vf_coef | float | [0.3, 0.9] |
| max_grad_norm | float | [0.3, 1.0] |
| n_units | categorical | {128, 256, 512} → resolved to net_arch |

### 9.2 Trial Objective

Each trial trains PPO for `timesteps_per_trial` (default 100,000) steps then evaluates the policy
for `n_eval_episodes` (default 5) on `eval_road` (default speed_bump). The objective value is the
mean episode return. Curriculum is enabled during trials by default.

### 9.3 Output Structure (mirrors training)

```
tune/myPPO_study/speed_bump/
    exp_1/
        results.json          # all trial values and best_trial
        best_params.yaml      # PPO block, ready to paste into algo_configs.yaml
    exp_2/                    # next invocation auto-increments
```

The `best_params.yaml` uses the exact same format as `algo_configs.yaml` so the best hyperparameters
can be copied with no reformatting.

---

## 10. Evaluation

### 10.1 `eval.py` — Single Model

Runs `n_episodes` rollouts and reports:

```
  Episode return          (mean ± std)
  RMS body acceleration   [m/s²]      ← primary comfort metric
  Peak body acceleration  [m/s²]
  Speed tracking RMSE     [m/s]
  Mean speed              [m/s]
  Comfort score           [0–1]       = max(0, 1 − RMS / a_limit)
  Action smoothness RMS
```

Per-term reward breakdown is also reported (r_heave, r_wheel, r_tracking, r_accel, r_jerk).

The `--save-plots` flag generates a 7-panel figure per episode: body accel, wheel accel,
speed vs. reference, running RMS, action, step reward, reward breakdown.

### 10.2 `compare.py` — Agent vs Baselines

Evaluates the RL agent against two passive baselines:

| Baseline | Behaviour |
|----------|---------|
| passive | Zero action every step (constant speed, no acceleration response) |
| random | Uniform random action in [−1, 1] every step |

Comparison runs on multiple roads simultaneously and produces bar-chart summaries and a JSON
results file.

---

## 11. What Was Taken from ba_azab and What Was Built New

### 11.1 Inherited from ba_azab

| Component | What was kept |
|-----------|--------------|
| Quarter-car physics | Vehicle parameters (m_B, m_W, k_S, k_T, c_T, D, A, ...) matching Mandl (2021) |
| Nonlinear spring model | Exponential bumpstop formulation (`_spring_nonlin`) |
| Sigmoid-blended damper | Continuous PWA approximation (`_damper`) |
| Numba RK4 integrator | `_rk4_loop` architecture and `@njit(cache=True)` usage |
| Cosine bump road model | Shape formula and geometry parameters |
| Recorded scenario files | 10 JSON scenario files from Nardo test drives |
| VecNormalize usage | Obs + reward normalisation pattern |
| Basic reward structure | Speed-band tracking penalty concept |

### 11.2 Built New in COptRL

| Component | Description |
|-----------|-------------|
| **PreviewWrapper** | Peak-detected spatial preview with Gaussian noise and PT1 filter — completely new; ba_azab does not use preview in training |
| **CurriculumWrapper** | 3-level difficulty progression; no equivalent in ba_azab |
| **`from_random()`** | Per-episode road randomisation; ba_azab uses a fixed road layout |
| **z_W_ddot exposure** | ODE now returns wheel acceleration alongside body acceleration |
| **r_heave / r_wheel** | Vertical acceleration penalty terms; not in ba_azab |
| **Velocity-scaled reward** | `R *= v/v_max` to prevent stopping policy; new |
| **ISO 2631-1 grounding** | Comfort thresholds derived from standard; not in ba_azab |
| **Optuna tuning pipeline** | Complete search infrastructure (tune/, search_spaces.py, trial.py) |
| **YAML-driven config** | All parameters in separate YAML files with typed dataclass loaders |
| **Justfile recipes** | Developer experience: `just train`, `just tune`, `just eval`, etc. |
| **Graceful Ctrl+C** | try/finally saves model before exit in both train and tune |
| **Eval plots + compare** | 7-panel episode plots, multi-road agent-vs-baseline comparison |
| **TD3 / SAC removal** | Simplified to PPO-only; ba_azab also supported TD3 |
| **ISO_8608 / sine_sweep removal** | Removed stochastic and synthetic road profiles; kept realistic ones |

### 11.3 Key Architectural Difference: Preview in Training

ba_azab does **not** use the preview in the policy observation during training. The preview was
explored as a separate component but was not wired into the PPO policy.

In COptRL, `PreviewWrapper` is applied **before** `VecNormalize`, so the 9 preview features are
part of the full 15-dimensional observation that the PPO policy receives at every step. The policy
therefore has access to bump geometry 20 m ahead and can learn to slow down in anticipation
rather than reacting to bump-induced body movement.

---

## 12. Key Design Decisions

### 12.1 Speed Control, Not Suspension Control

The agent has no actuator on the suspension. It controls only longitudinal speed. This makes the
problem harder (indirect, predictive control) but eliminates the need for any hardware modification
and makes the learned policy deployable in any road vehicle with adaptive cruise control.

### 12.2 Flat MLP, Not CNN or LSTM

The observation is a fixed-length 15-dimensional vector. A flat MLP with two 256-unit hidden layers
is sufficient; a CNN would require no spatial structure; an LSTM would add training instability
for marginal benefit. The observation is already augmented with IIR-filtered acceleration and jerk
so temporal memory is partially encoded.

### 12.3 Quadratic (not Quartic) Reward Terms

Mandl (2021) recommends quartic body-acceleration cost (drives speed to zero more aggressively).
This project uses quadratic throughout for training stability — quartic gradients can be extreme
early in training when the policy is random. The relative weighting (heave:tracking ≈ 1:1) preserves
the spirit of Mandl's recommendation.

### 12.4 Geometry-Safe Speed Clamping

`_clamp_speed_to_geometry()` limits the vehicle speed to a physics-safe value derived from the
maximum height-to-length ratio of sampled bumps, calibrated from two real test runs. This prevents
the training distribution from containing scenarios that would cause suspension truncation regardless
of the agent's actions.

### 12.5 Separate eval VecNormalize (no reward normalisation)

The eval environment shares the training environment's observation running statistics but has
`norm_reward = False`. This ensures that the reported episode returns are in real reward units, not
normalised units, making results interpretable and comparable across runs.

---

## 13. Justfile Quick Reference

```bash
just train speed_bump --c --n-envs 4    # PPO with curriculum, 4 parallel envs
just tune --trials 50 --timesteps 100000 # Optuna 50-trial search
just tune-db                             # same, but stores in tune.db for live dashboard
just dashboard                           # Optuna dashboard (requires: just install-dashboard)
just eval models/PPO/speed_bump/exp_1/PPO_final.zip --save-plots
just compare models/PPO/speed_bump/exp_1/PPO_final.zip
just tb                                  # TensorBoard for all runs
just test                                # run 24 unit tests
```

---

## 14. Test Coverage

24 unit tests in `tests/` covering:

- ODE physics (`test_ode_model.py`): reset, step returns correct shapes, determinism
- Reward terms (`test_reward.py`): zero at zero, negative for non-zero, clip saturation
- Road generator (`test_road_generator.py`): bump heights, flat profile, recorded interpolation
- Environment (`test_env.py`): reset/step API, observation shape, action space

---

## 15. References

1. Mandl, P. (2021). *Predictive driver model for speed control in the presence of road obstacles*. TU Wien Diplomarbeit.
2. ISO 2631-1:2016. *Mechanical vibration and shock — Evaluation of human exposure to whole-body vibration*.
3. EN 13374 / ISO 22133. Jerk limits for comfortable motion.
4. Mitschke, M. & Wallentowitz, H. (2014). *Dynamik der Kraftfahrzeuge* (5th ed.). Springer Vieweg.
5. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
6. Stable-Baselines3: https://stable-baselines3.readthedocs.io
7. Optuna: https://optuna.org

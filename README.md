# COptRL: Comfort-Optimising Reinforcement Learning

**Can a car learn to plan its own speed ahead before a speed bump so the ride doesn't feel awful?**

That's the question. No fancy active suspension, no hydraulics, just a policy that looks ahead, decides how fast to approach each bump, and keeps passengers comfortable using only the throttle and brake.

![Human driver episode trace](eval/results/human_driver/speed_bump/human_driver_ep1.gif)

*Human driver baseline navigating a random speed-bump road. Speed (blue) drops ahead of each bump; body acceleration (orange) spikes show what still gets through.*

---

## What is this?

A reinforcement learning agent trained to plan longitudinal vehicle speed in a quarter-car simulation. The agent sees upcoming road bumps via a peak-detected preview, observes filtered suspension signals, and outputs a normalised acceleration command every 20 ms. The goal: cross bumps fast and smooth.

The environment runs a full nonlinear quarter-car ODE (Numba-jitted RK4 at 1 ms steps) wrapped in a Gymnasium interface. Training uses PPO with a staged performance-gated curriculum and per-episode road randomisation.

**Baselines included:** rule-based human driver and an acados MPC controller with a 150-step receding horizon (3.0 s).

---

## Results at a glance

| Method | Mean return | RMS body accel | Comfort score | Notes |
|---|---|---|---|---|
| PPO | **-** | — | — | exp_16, 1.84M steps, level 0 mastered |
| Human driver | −644 | 5.78 m/s² | 0.0 | kinematic look-ahead planner |
| MPC (acados) | — | 3.66 m/s² | 0.34 | not re-evaluated on Mandl reward scale |
| Constant speed | — | high | 0.0 | not re-evaluated on Mandl reward scale |

Returns measured over randomised speed-bump episodes under Mandl (2024) reward weights (Q_zBddot=50, g-normalised). Lower = worse. MPC/constant-speed returns are pending re-evaluation on the new reward scale.

![Human driver episode 1: speed and body acceleration](eval/results/human_driver/speed_bump/human_driver_ep1.png)

*Ep 1: four bumps. The human driver brakes consistently but the fast approach leaves peaks of ±30 m/s² at the bumps it doesn't fully scrub off.*

![Human driver episode 5: cleaner single-bump crossing](eval/results/human_driver/speed_bump/human_driver_ep5.png)

*Ep 5: one heavy bump at t ≈ 2 s. Smooth braking approach, still a ±30 m/s² spike on entry.*

---

## Quick start

```bash
# install
just install

# train PPO with curriculum (4 parallel envs, 3M steps)
just train speed_bump --curriculum --n-envs 4

# evaluate best checkpoint
just eval models/PPO/speed_bump/exp_16/best/best_model.zip --save-plots

# compare RL vs MPC vs human driver
just compare models/PPO/speed_bump/exp_16/best/best_model.zip

# run human driver baseline : 20 episodes, save GIF + plots
just human-driver --n-episodes 20 --save-gif --save-plots

# run MPC baseline (requires acados)
just mpc --n-episodes 20 --save-plots

# TensorBoard
just tb

# Optuna hyperparameter search (50 trials)
just tune --trials 50

# live Optuna dashboard
just dashboard
```

---

## The environment

### Quarter-car model

A 2-DOF nonlinear quarter-car. The suspension spring includes an exponential bumpstop progression; the damper has a sigmoid-blended asymmetric piecewise-linear characteristic (separate compression and rebound slopes).

```
m_B · z̈_B  =  −F_spring(z_W − z_B) − F_damp(ż_W − ż_B)
m_W · z̈_W  =   F_spring + F_damp − k_T(z_W − ζ) − c_T(ż_W − ζ̇)
```

| Parameter | Value | What it is |
|---|---|---|
| m_B | 465.7 kg | sprung (body) mass |
| m_W | 50.4 kg | unsprung (wheel) mass |
| k_S | 27 922 N/m | suspension spring stiffness |
| k_T | 262 200 N/m | tyre radial stiffness |
| c_T | 500 N·s/m | tyre damping |

Integration: RK4, 20 sub-steps per 20 ms control interval (effective 1 ms physics step). All kernels are Numba-jitted : a 2000-step episode runs in under 30 ms on CPU.

### State vector

| Index | Variable | Description |
|---|---|---|
| x[0] | ζ − z_W | tyre deflection [m] |
| x[1] | ż_W | wheel vertical velocity [m/s] |
| x[2] | z_W − z_B | suspension travel [m] |
| x[3] | ż_B | body vertical velocity [m/s] |
| x[4] | v | longitudinal speed [m/s] — set by agent |
| x[5] | z_B | body displacement [m] |

### Action and observation

**Action:** scalar float32 ∈ [−1, 1] → scaled to `a_cmd = u × a_max` [m/s²].

**Observation:** 7 base signals + 3×n_peaks from `PreviewWrapper`.

| Index | Signal | Range |
|---|---|---|
| 0 | road height at wheel ζ | ±0.15 m |
| 1 | road velocity at wheel ζ̇ | ±7.0 m/s |
| 2 | v / v_max | [0, 1] |
| 3 | filtered longitudinal accel / a_comfort | IIR α=0.8 |
| 4 | filtered jerk / j_max | IIR α=0.8 |
| 5 | prev_action | [−1, 1] |
| 6 | v_init / v_max | [0, 1] |
| 7… | bump preview (t2r, height, freq) per peak | [0, 1] each |

The preview samples 200 points over a 60 m lookahead, runs `scipy.signal.find_peaks`, and encodes up to *n* bumps as `[t2r/T_MAX, h/h_clip, freq/_FREQ_MAX]` where `t2r = dist/v` (time-to-reach) and `freq = v/width` (crossing frequency). Missing peaks fill with `[1.0, 0.0, 0.0]`. PT1 filter (τ = 0.05 s) smooths the output.

---

## Reward function

The per-step reward follows Mandl (2024) with g-normalised quadratic cost terms:

```
R = (v/v_max) × [Q_zBddot·J_heave + Q_zWddot·J_wheel + Q_a·J_long]   (velocity-scaled)
  + Q_v · J_speed                              (unscaled — always costs the same)
  + w_jerk · J_jerk + w_action_smooth · J_smooth  (unscaled — see WHY_WE_DO_THAT.md)
  + w_progress · J_progress
  + Q_step
  + w_bump_cross  (one-shot per bump cleared)
```

**Velocity scaling** on the comfort terms means going slow doesn't help — slower speed = smaller multiplier, so the agent can't farm a comfort score by crawling.

**Tracking and smoothness are unscaled** — they cost the same at any speed. Without this, the agent learned to drive slowly and oscillate freely (see `WHY_WE_DO_THAT.md`).

**`J_progress = s_pos / road_length`** is always positive — it's what stops the agent from sitting still before every bump.

**Bump-crossing bonus** (`w_bump_cross = 20.0`) fires once each time the vehicle clears the end of a bump.

| Term | Formula | Role |
|---|---|---|
| J_heave | −(clip(z̈_B, ±8) / g)² | vertical body accel, g-normalised (Mandl 2024) |
| J_wheel | −(clip(z̈_W, ±60) / g)² | wheel-hop / road holding |
| J_speed | −\|v−v_init\|/v_init above v_min; −1−((v_min−v)/v_min)² below | absolute speed tracking |
| J_long | −(clip(ā, ±4) / g)² | longitudinal ride comfort |
| J_jerk | −(clip(j̄, ±4) / j_max)² | smoothness of acceleration |
| J_smooth | −(u_t − u_{t−1})² | control chatter |
| J_progress | s_pos / road_length | positive reward for forward progress |
| bump bonus | +w_bump_cross | one-shot per bump end cleared |

Terminal bonus of ±100 at episode end — requires **both** low RMS body accel (< a_limit = 5 m/s²) and a minimum mean speed.

Current weights: `Q_zBddot=50.0, Q_zWddot=0.5, Q_a=1.0, Q_v=1.0, Q_step=0.1, w_jerk=0.4, w_action_smooth=0.1, w_progress=0.15, w_bump_cross=20.0`

---

## Training

**Algorithm:** PPO (`MlpPolicy`, net_arch [128, 128] for both actor and critic), with `VecNormalize` on observations and rewards.

**Curriculum:** 4 performance-gated levels. The agent must sustain mean eval return above a threshold for 5 consecutive evaluations before advancing. This stops it from being pushed to harder roads before it's ready.

| Level | Bump catalog IDs | Gap range | Advance threshold |
|---|---|---|---|
| 0 (easy) | 3, 4 — long/gentle slopes | wide | 0 |
| 1 (medium) | 0, 1, 3, 4 | medium | −100 |
| 2 (hard) | 0–4, incl. severe | tight | −300 |
| 3 (expert) | 0–4, more bumps | very tight | — |

**Speed bump catalog** (from Mandl 2021 physical measurements):

| ID | Name | Height | Width | Peak ζ̇ at 20 m/s |
|---|---|---|---|---|
| 0 | short bump | 2.5 cm | 0.92 m | 1.71 m/s |
| 1 | medium bump | 6.25 cm | 2.22 m | 1.77 m/s |
| 2 | severe bump | 10 cm | 1.00 m | 6.28 m/s |
| 3 | long bump | 12.5 cm | 9.50 m | 0.83 m/s |
| 4 | raised crosswalk | 10 cm | 5.00 m | 1.26 m/s |

**Best run (exp_16):** 3M steps, 4 envs, curriculum, seed 69. Best checkpoint at 1.84M steps: mean eval return **−25**. Level 0 mastered at 1.86M steps, level 1 still training when run ended.

---

## Baselines

### Human driver

A kinematic look-ahead planner. It scans ahead up to 40 m for bumps, computes the fastest safe crossing speed from the bump slope (π·H/W), calculates the braking distance, and ramps speed down if inside that zone. Pure geometry, no model, no optimisation.

```bash
just human-driver --n-episodes 20 --save-gif --save-plots
just human-driver --scenario single_severe
```

### MPC (acados)

Model Predictive Control with a 50-step receding horizon (1 s lookahead) at 20 ms intervals. Full nonlinear quarter-car ODE as prediction model, solved with SQP-RTI + HPIPM. Solve time: ~1–2 ms per step.

```bash
just mpc --n-episodes 20 --save-plots
just mpc --scenario urban_gauntlet
```

Solver is rebuilt and recompiled on first encounter with each unique bump geometry, then cached by a hash of the road layout.

---

## Eval scenarios

Named scenarios in `config/eval/scenarios/` fix a bump layout and vehicle speed for reproducible comparison. All use bumps from the Mandl 2021 catalog.

| Scenario | Bumps | Speed | Description |
|---|---|---|---|
| single_severe | 1 × severe | 30 km/h | worst-case single bump |
| urban_gauntlet | 5 mixed | 30 km/h | varied bump types in sequence |
| highway_long | 2 × long | 50 km/h | gentle long bumps at speed |
| *(see config/eval/scenarios/)* | | | |

```bash
just human-driver --scenario urban_gauntlet --save-plots
just mpc --scenario urban_gauntlet --save-plots
```

---

## Configuration

All config lives in `config/` as YAML. No reinstall needed, changes take effect on the next run. CLI flags always override YAML values.

| File | Controls |
|---|---|
| `config/algo/algo_configs.yaml` | PPO hyperparameters, training budget, n_envs |
| `config/algo/tune_config.yaml` | Optuna search space |
| `config/reward/reward_params.yaml` | comfort weights and ISO 2631 thresholds |
| `config/curriculum/curriculum_params.yaml` | level thresholds and bump IDs per level |
| `config/baseline/mpc_params.yaml` | MPC horizon, solver settings |
| `config/baseline/human_driver_params.yaml` | preview distance, brake decel |
| `config/road/speed_bumps.json` | physical bump catalog |

See `config/README.md` for full parameter documentation.

---

## Why speed planning instead of active suspension?

Active suspension reacts to a bump after the wheel hits it. There's always actuator latency, typically 10–50 ms, so some vibration gets through regardless. If the car slows down *before* the bump, the passive suspension handles it within comfort limits without any actuator at all. The only hardware needed is GPS/camera preview of the road.

This isn't a replacement for active suspension on serious roads, it's the free comfort improvement you can get from smarter speed management. See `WHY_WE_DO_THAT.md` for design decisions and `REPORT.md` for the full technical write-up.
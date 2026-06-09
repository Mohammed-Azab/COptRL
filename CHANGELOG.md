# Changelog

## exp_21 results and reward/curriculum calibration

exp_21: 5M steps, PPO with Optuna-tuned hyperparameters, curriculum on, step_bonus=0.5.
Agent stayed on level 0 the entire run — never advanced. Best model saved at 80k steps.
Eval return oscillated +145–+212 for 4.5M steps with no improvement trend.

Key findings:
- **Curriculum threshold too high**: level 0 threshold was +230 but agent's rolling-10 eval mean peaked at ~+198. Threshold was above the observable ceiling, making advancement impossible.
- **Early plateau**: policy converged by ~500k steps; remaining 4.5M steps added nothing. The Optuna hyperparameters (clip_range=0.1, lin_3e-5 LR decay) learn correctly but conservatively — the step budget was effectively wasted after the agent reached its skill ceiling on level 0.

Changes:
- Lowered thresholds so agent can actually advance: +230/+210/+190 → **+190/+175/+155**
- Added `step_bonus=0.5` to reward: constant +0.5 added to every step, shifting episode returns by ~+150. Human driver mean: −123 → +19.8 (barely positive = "neutral"). Good RL returns: +50 → +200. Bad returns stay negative.
- `reward_bounds()` in `utils.py` now includes `step_bonus` so printed range reflects the actual reward scale.

Baselines after step_bonus:
- Human driver (random road, 100 ep): mean +24.7, max +294, std 142
- Fixed scenarios: single_crosswalk +203.7, urban_gauntlet +318.7, table31_resonance +334.5

---

## PPO hyperparameter tuning for level 3 stability

exp_20 showed the agent finding good policies at level 3 (+27.5 best) but immediately bouncing away — a classic sign that the constant learning rate is too large for fine-tuning after curriculum advances.

- **Linear LR decay**: `3e-4 → 0` over the training run. LR at level 3 entry (~720k steps) is still ~2.6e-4 for fast learning; by 4M steps it has fallen to ~6e-5 for fine-tuning.
- **Larger rollouts**: `n_steps` 2048 → 4096, `batch_size` 64 → 256. More environment steps per update reduces gradient variance without increasing wall time.
- **Entropy coefficient**: `ent_coef` 0.0 → 0.005. A small entropy bonus prevents premature policy collapse when the agent overshoots and needs to recover.
- **Eval episodes**: 5 → 10 per evaluation so the mean used for curriculum advancement gating is more reliable.

---

## Curriculum and training improvements

Raised curriculum advancement thresholds from negative values to positive returns so the agent must genuinely master each level before moving up. exp_19 showed the agent blowing through levels 0–2 in 100k steps each because thresholds (-80/-60/-40) were trivially easy; level 3 then ran 2.7M steps without convergence.

- Level 0 threshold: −80 → +50, level 1: −60 → +30, level 2: −40 → +10
- Total timesteps: 3M → 5M to match the longer per-level budget
- Per-level best model saving: training now saves `best/level_N/best_model.zip` alongside the existing overall best whenever a new per-level eval return is achieved
- Fixed misleading `(+X from max)` label in training summary → `(+X below ceiling)` so it's clear the number is the gap to the theoretical max, not the max itself

---

## Reward config fixes and bumps display

- `w_bump_cross`: 50 → 10 — one-shot bonus was 250× larger than the per-step max, causing wild oscillation in exp_17
- Synced `RewardConfig` dataclass defaults to match the YAML values (stale defaults were silently used if YAML loading failed)
- `bumps_passed/total` now shown in `eval.py`, `driver_eval.py`, and `mpc.py` (e.g. `3/4`)
- Documented all reward calibration lessons in `trial_error_reward.txt`

---

## MPC stop-and-wait fix

MPC was creeping at v_min and never reaching bumps (0/2 bumps passed). Three root causes fixed:

- **Velocity scaling in cost** — heave, wheel, and longitudinal-accel residuals are now multiplied by `v/v_max`, mirroring the env reward. Creeping at v_min no longer makes comfort violations negligibly cheap.
- **Horizon N: 50 → 150** — 3-second lookahead covers braking distance + bump + recovery. With partial condensing (`cond_N=10`) solve time stays ~3 ms.
- **Step budget** — episode steps now based on `v_min` as worst-case speed, ensuring the car has enough steps to reach every bump even if it moves slowly.

---

## Suppress native solver output during MPC execution

Added a context manager that captures stdout/stderr at the file-descriptor level so acados doesn't dump C-level solver messages to the terminal during episode rollouts. Makes the eval output readable when running many episodes back to back.

---

## Render improvements

**Dashed bump markers on time-series plots** — vertical dashed lines now mark each bump crossing on the speed, accel, and action time-series panels. Controlled by `RENDER_BUMP_MARKERS` in render config.

**Full episode history** — time-series panels now show the complete episode trace from step 0 instead of a sliding window. Makes it easier to read what happened across the whole run.

**Render accuracy and episode reset fixes** — fixed a bug where render state could carry over between episodes, causing ghost traces. Episode budget and road length are now recalculated correctly when the env resets.

---

## Human driver baseline and evaluation

Added a rule-based human driver in `src/baseline/human_driver/`. It scans ahead for bumps and brakes with a kinematic ramp to a safe crossing speed, then re-accelerates. A companion eval script mirrors the MPC evaluation structure.

- Fixed braking logic: driver now brakes aggressively inside the braking zone instead of relying on a P-controller that goes near-zero at onset
- Wired both baselines (MPC and human driver) to their respective config YAML files
- Added GIF saving and per-episode time-series plots to both eval scripts
- Fixed PYTHONPATH so `scenario_loader` is importable when running evals from the baseline dirs

---

## Mandl eval scenarios

Added 9 named evaluation scenarios in `config/eval/scenarios/` drawn from Mandl (2021) bump catalog. Each scenario fixes the bump layout, positions, and vehicle speed so MPC and RL results are directly comparable. Bump lead-in distance increased from 12 m to 60 m to give the car enough room to brake before the first bump.

---

## Reward calibration for Mandl bump catalog

Tuned `w_tracking`, `w_heave`, `w_wheel`, `a_B_comfort`, and `a_W_comfort` against the Mandl bump catalog so reward magnitudes are consistent across difficulty levels.

---

## Performance-gated curriculum

Curriculum no longer advances on a fixed step schedule. The agent must sustain a mean eval return above the level threshold for 5 consecutive evaluations before moving up. This stops it from advancing before it has actually learned the current level.

- 4 levels (easy → expert), each with a configurable return threshold
- Level selection controls eligible bump catalog IDs and gap ranges
- Documented in `WHY_WE_DO_THAT.md` (Issue 12)

---

## r_progress reward term

Added `r_progress = v / v_max` as a small positive encouragement for forward movement. This replaced the separate tracking penalty at low speed that was causing a stop-and-wait exploit. Episode return range is now roughly −600 to +160.

---

## Fix jerk and action-smooth exploit

`r_jerk` and `r_action_smooth` were velocity-scaled, which made them near-zero at low speed. The agent learned to slow down to suppress these penalties. Fixed by removing the velocity scaling from both terms — they now penalise any jerk regardless of speed. Documented in `WHY_WE_DO_THAT.md` (Issue 5).

---

## Road and geometry fixes

- Fixed road-position drift accumulating over long episodes
- Geometry clamp now uses physics-based limits instead of arbitrary bounds
- `TRUNC_TRAVEL` (suspension over-travel truncation) calibrated to 0.20 m
- `road_complete` now correctly triggers termination (terminal bonus fires) rather than truncation

---

## Reward calibration and bounds

- `w_tracking` dropped from 0.8 to 0.3 after removing the dead band made the tracking term always-active
- Removed a double-counted tracking term from the reward bounds calculation
- Dead band removed from `r_tracking` so it penalises any distance from `v_max`, not just outside a window (fixes creep exploit)

---

## Hyperparameter tuning with Optuna

Added `src/tune/` with an Optuna-based PPO search. Search space is defined in `config/algo/tune_config.yaml`. Results are stored as YAML in `tune/results/` and the best params feed back into training. Dashboard available via `optuna-dashboard`.

---

## Mandl 2021 speed bump catalog with realistic dimensions

**Speed bumps** are now drawn from a catalog of 5 physically measured bump types taken from Mandl (2021), stored in `config/road/speed_bumps.json`. Previously, bump height and width were sampled from arbitrary uniform ranges with no physical grounding.

Catalog entries (0-based ID):

| ID | Name | Height | Width | Source |
|----|------|--------|-------|--------|
| 0 | short_bump | 2.5 cm | 0.92 m | Table 3.1 |
| 1 | medium_bump | 6.25 cm | 2.22 m | Table 3.1 |
| 2 | severe_bump | 10 cm | 1.00 m | §5.1.1 |
| 3 | long_bump | 12.5 cm | 9.50 m | Table 3.1 |
| 4 | raised_crosswalk | 10 cm | 5.00 m | §5.1.2 |

**Curriculum** levels now select eligible bump types by catalog ID rather than random height/width ranges. Difficulty is ordered by peak road velocity excitation `ζ̇_max = π·H/W·v`:

- Level 0 (easy): IDs 3, 4 — gentle slopes, peak ζ̇ < 1.5 m/s at 20 m/s
- Level 1 (medium): IDs 0, 1, 3, 4
- Level 2 (hard): IDs 0–4, introduces severe bump
- Level 3 (expert): IDs 0–4, more bumps, tighter gaps

Advancement is performance-gated: the agent must sustain a mean eval return above the level threshold for 5 consecutive evaluations before moving up.

**`w_bump_cross`** updated to 5.0; reward fires once per bump crossing.

---

## MPC baseline with fixed speed tracking

Added a model predictive control (MPC) baseline in `src/baseline/mpc/` using acados and CasADi. The OCP models quarter-car dynamics over a 50-step horizon (dt = 20 ms) with the full nonlinear suspension, bumpstop, and piecewise damper.

**Bug fixed — one-sided speed tracking:** The original cost used `fmax(0, v_ref − v)`, which was zero whenever the vehicle exceeded v_ref. This gave the solver no incentive to brake before a bump; it only paid a cost once already on top of it. Fixed by using the symmetric residual `(v − v_ref) / v_max` with target 0, so overspeed and underspeed are penalised equally.

**Progress residual removed:** A separate `v/v_max` term with target 1.0 was removed because it always pulled toward maximum speed even when `v_ref` was set lower ahead of a bump, partially cancelling the tracking fix. Since `_v_ref_at` already returns `v_max` on flat sections, the tracking term alone provides the "go fast" incentive.

Performance context: constant zero-throttle scores ~−359; MPC with the original bug scored ~−286; RL scores ~−25.

---

## Scale up training and clean up code

- `n_envs` 1 → 4, `total_timesteps` 1M → 3M, `checkpoint_freq` 50k → 100k
- Converted multi-line docstrings to short inline comments in `ode_model.py`, `environment.py`, `monitoring.py`
- Removed `acados/` (external compiled dependency) and local-only markdown files from git tracking


**Speed bumps** are now drawn from a catalog of 5 physically measured bump types taken from Mandl (2021), stored in `config/road/speed_bumps.json`. Previously, bump height and width were sampled from arbitrary uniform ranges with no physical grounding.

Catalog entries (0-based ID):

| ID | Name | Height | Width | Source |
|----|------|--------|-------|--------|
| 0 | short_bump | 2.5 cm | 0.92 m | Table 3.1 |
| 1 | medium_bump | 6.25 cm | 2.22 m | Table 3.1 |
| 2 | severe_bump | 10 cm | 1.00 m | §5.1.1 |
| 3 | long_bump | 12.5 cm | 9.50 m | Table 3.1 |
| 4 | raised_crosswalk | 10 cm | 5.00 m | §5.1.2 |

**Curriculum** levels now select eligible bump types by catalog ID rather than random height/width ranges. Difficulty is ordered by peak road velocity excitation `ζ̇_max = π·H/W·v`:

- Level 0 (easy): IDs 3, 4 — gentle slopes, peak ζ̇ < 1.5 m/s at 20 m/s
- Level 1 (medium): IDs 0, 1, 3, 4
- Level 2 (hard): IDs 0–4, introduces severe bump
- Level 3 (expert): IDs 0–4, more bumps, tighter gaps

Advancement is performance-gated: the agent must sustain a mean eval return above the level threshold for 5 consecutive evaluations before moving up.

**`w_bump_cross`** updated to 5.0; reward fires once per bump crossing.

---

## MPC baseline with fixed speed tracking

Added a model predictive control (MPC) baseline in `src/baseline/mpc/` using acados and CasADi. The OCP models quarter-car dynamics over a 50-step horizon (dt = 20 ms) with the full nonlinear suspension, bumpstop, and piecewise damper.

**Bug fixed — one-sided speed tracking:** The original cost used `fmax(0, v_ref − v)`, which was zero whenever the vehicle exceeded v_ref. This gave the solver no incentive to brake before a bump; it only paid a cost once already on top of it. Fixed by using the symmetric residual `(v − v_ref) / v_max` with target 0, so overspeed and underspeed are penalised equally.

**Progress residual removed:** A separate `v/v_max` term with target 1.0 was removed because it always pulled toward maximum speed even when `v_ref` was set lower ahead of a bump, partially cancelling the tracking fix. Since `_v_ref_at` already returns `v_max` on flat sections, the tracking term alone provides the "go fast" incentive.

Performance context: constant zero-throttle scores ~−359; MPC with the original bug scored ~−286; RL scores ~−25.

---

## Scale up training and clean up code

- `n_envs` 1 → 4, `total_timesteps` 1M → 3M, `checkpoint_freq` 50k → 100k
- Converted multi-line docstrings to short inline comments in `ode_model.py`, `environment.py`, `monitoring.py`
- Removed `acados/` (external compiled dependency) and local-only markdown files from git tracking

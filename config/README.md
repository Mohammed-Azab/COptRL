# Configuration Reference

All YAML files in this directory are loaded at runtime. CLI flags always override YAML values. Changes take effect immediately — no reinstall needed.

---

## Bump catalog (Mandl 2021)

Referenced by 0-based `catalog_id` in road, curriculum, and scenario configs. Dimensions from `src/road/speed_bumps.json`. Peak ζ̇ = π·H/W·v.

| ID | Name | H (cm) | W (m) | Peak ζ̇ @ 20 m/s |
|----|------|---------|-------|-----------------|
| 0  | short_bump | 2.5 | 0.92 | ~1.7 m/s |
| 1  | medium_bump | 6.25 | 2.22 | ~1.8 m/s |
| 2  | severe_bump | 10 | 1.00 | ~6.3 m/s |
| 3  | long_bump | 12.5 | 9.50 | ~0.8 m/s |
| 4  | raised_crosswalk | 10 | 5.00 | ~1.3 m/s |

---

## `algo/algo_configs.yaml`

PPO constructor kwargs passed directly to SB3, plus training meta-settings consumed by `train.py`.

```
PPO:
  learning_rate      learning rate for Adam optimiser
  n_steps            rollout buffer size per env per update
  batch_size         mini-batch size (must divide n_steps × n_envs)
  n_epochs           gradient passes over each rollout buffer
  gamma              discount factor
  gae_lambda         GAE advantage estimation parameter
  clip_range         PPO policy clipping (ε)
  ent_coef           entropy bonus coefficient (0 = off)
  vf_coef            value function loss weight
  max_grad_norm      gradient clipping norm
  policy_kwargs:
    net_arch:
      pi: [256, 256]  actor hidden layers
      vf: [256, 256]  critic hidden layers

training:
  seed               base random seed
  total_timesteps    default training budget
  n_envs             parallel environments
  n_eval_envs        parallel eval environments
  eval_freq          env steps between eval callbacks
  n_eval_episodes    episodes per eval callback
  checkpoint_freq    env steps between checkpoint saves
  norm_obs           enable VecNormalize on observations
  norm_reward        enable VecNormalize on rewards (on-policy only)
  eval_road          road profile used for eval callbacks
```

---

## `algo/tune_config.yaml`

Optuna PPO search space and tuning defaults. Each parameter entry has a `type` and either `low`/`high` (for continuous) or `choices` (for categorical). `n_units` is a synthetic alias resolved to `policy_kwargs.net_arch` by the sampler — it does not map directly to a PPO kwarg.

```
PPO:
  learning_rate      float_log  [1e-5, 1e-3]
  n_steps            categorical [2048, 4096, 8192]
  batch_size         categorical [64, 128, 256]
  n_epochs           int        [5, 20]
  gamma              float      [0.99, 0.9999]
  gae_lambda         float      [0.90, 0.99]
  clip_range         categorical [0.1, 0.2, 0.3]
  ent_coef           float_log  [1e-8, 1e-2]
  vf_coef            float      [0.3, 0.9]
  max_grad_norm      float      [0.3, 1.0]
  n_units            categorical [128, 256, 512]

defaults:
  timesteps_per_trial   env steps each trial trains for
  n_eval_episodes       episodes per trial evaluation (keep low for speed)
  eval_road / train_road
  seed
  use_curriculum        whether trials use CurriculumWrapper
```

After a tuning run, `tune/<study_name>/<road>/exp_<n>/best_params.yaml` is written in the same format as `algo_configs.yaml` and can be copied directly into that file.

### Tuning history
- `learning_rate`: using `lin_3e-4` (linear decay to 0) — a fixed rate of 8.49e-05 from an Optuna trial degraded after curriculum level advances.
- `net_arch pi/vf`: [256, 256] — [512, 512] showed no benefit.
- `ent_coef`: 0.005 — raised from 0 to prevent premature policy collapse.
- `n_steps` / `batch_size`: 4096 / 256 — scaled together to reduce gradient variance.

---

## `reward/reward_params.yaml`

Controls all reward terms. See `src/gym_env/QuarterCar_env/reward/reward.py` for the exact formulae.

```
weights:
  w_tracking        weight on speed-band penalty
  w_accel           weight on longitudinal accel penalty
  w_jerk            weight on jerk penalty
  w_action_smooth   weight on action-change penalty

vertical:
  w_heave           weight on body vertical accel penalty
  w_wheel           weight on wheel vertical accel penalty
  a_B_comfort       body accel normaliser [m/s²]  (ISO 2631 reference)
  a_W_comfort       wheel accel normaliser [m/s²]
  enable_heave / enable_wheel / enable_vel_scaling   feature flags

enable:             per-term on/off flags for longitudinal terms

velocity:
  v_max             target/upper speed [m/s]
  a_max             maximum acceleration command magnitude [m/s²]
  v_min             lower speed bound; stopping is penalised below this

comfort:
  a_comfort         longitudinal accel normaliser [m/s²]
  accel_filter_alpha  IIR smoothing (0 = no filter, →1 = heavy)
  accel_clip        observation clip for filtered accel [m/s²]
  reward_accel_clip reward-only clip (set to 2 × a_comfort)

jerk:               same structure as comfort, for jerk [m/s³]

terminal:
  terminal_bonus    reward at episode end if RMS body accel < a_limit
  terminal_penalty  reward at episode end if RMS body accel ≥ a_limit
  a_limit           RMS threshold [m/s²]

observations:
  preview_distance    lookahead horizon [m]
  h_clip              height normalisation clip [m]
  n_peaks             max bumps encoded in the preview observation
  peak_height_min     minimum height to detect as a peak [m]
  peak_distance_min_m minimum separation between detected peaks [m]
  noise_active        add Gaussian noise to peak slots
  noise_height_std / noise_distance_std / noise_width_std
  pt1_tau             PT1 filter time constant [s]
```

### Calibration notes
- `a_B_comfort`: 3.0 (was 0.5) — normalises the heave reward so typical Mandl body accelerations (2–5 m/s²) map near 1.
- `reward_heave_clip`: 8.0 (was 1.0) — old value clipped too aggressively and killed the gradient.
- `a_limit` (terminal): 5.0 (was 1.0) — 5 m/s² RMS is achievable with good speed management.
- `preview_distance`: 60 m — matches the flat_start so the agent always sees the first bump as it starts moving.

---

## `road/road_params.yaml`

Fixed bump layout (used when `random_road_on_reset=False`) and random road bounds.

```
dis_mode            constant | custom — gap mode between bumps
num_bumps           number of bumps in the fixed layout
bump_sequence       ordered list of bump type indices
custom_dis          per-gap distances [m] (used when dis_mode=custom)
constant_dis        uniform gap [m] (used when dis_mode=constant)
bump_x_start        x-position of first bump [m]
bump_types:
  <id>:
    bump_height     amplitude [m]
    bump_length     wavelength [m]
vehicle_speed       default entry speed [m/s]

random:             bounds for RoadGenerator.from_random()
  num_bumps_range
  bump_height_range
  bump_length_range
  min_gap           minimum gap between consecutive bumps [m]
  flat_start        x-position of the first random bump [m]
  v_random_low_factor  v_random_low = v_min × this factor
```

---

## `curriculum/curriculum_params.yaml`

Difficulty levels for `CurriculumWrapper`. Thresholds are in per-env steps.

```
thresholds: [200_000, 500_000]   steps at which level 1 and level 2 unlock

levels:
  0:                             easiest — used from step 0
    num_bumps_range
    bump_height_range
    bump_length_range
    min_gap
    flat_start
    v_random_low / v_random_high  speed sampling range [m/s]
  1: ...
  2: ...                         hardest — unlocked after thresholds[1]
```

Add more levels by extending the `levels` dict and the `thresholds` list. The wrapper always uses the last level once all thresholds are passed.

### Threshold rationale
Thresholds require positive returns (+50 / +30 / +10). Earlier runs (exp_19) used negative thresholds (−80/−60/−40), which let the agent blow through levels 0–2 in ~100 k steps each without mastering them — level-2 mean was only −35 at advancement. Level 3 then ran 2.7 M steps with no convergence. `advance_window` is 5 (was 3) to require sustained performance before advancing.

---

## `gym_env/env_params.yaml`

Physics constants and episode settings. Changing these values requires reinstalling the gym package (`just build-gym-env`).

```
...
TRUNC_TRAVEL        suspension travel truncation limit [m]
TRUNC_ZS            body displacement truncation limit [m]
...
```

Truncation fires only on numerical blow-up, not normal operation. Limits calibrated from 50 worst-case (full-throttle) episodes: max |z_W − z_B| observed = 0.094 m, max |z_B| observed = 0.284 m.

---

## `baseline/mpc_params.yaml`

```
N                   prediction horizon [steps]  (N × DT = horizon in seconds)
nlp_solver_max_iter max SQP iterations per solve
n_episodes          default number of eval episodes
```

N = 150 steps at DT = 0.02 s gives a 3 s horizon, covering the full braking distance at 50 km/h (~40 m) plus bump crossing and recovery. With partial condensing (`cond_N = 10`) the QP size is fixed and solve time stays ~1 ms. `nlp_solver_max_iter = 10` gives good quality at ~5 ms/solve.

---

## `gym_env/render_params.yaml`

Matplotlib render geometry and colours. Only relevant when `--render` is passed to training or evaluation. These are cosmetic — changing them has no effect on training.

---

## `eval/compare_config.yaml`

Default settings for `compare.py`. All keys can be overridden via CLI flags.

```
algo            PPO
model_path      path to trained model .zip (required)
vecnorm_path    path to vecnormalize.pkl (null = auto-inferred)
roads           list of road profiles to evaluate
n_episodes      episodes per (agent × road) pair
deterministic   use deterministic policy (true) or sample (false)
seed            base random seed
baselines       list of baseline policies: passive | random
results_dir     output directory for JSON and plots
render          enable live render
save_plots      save matplotlib figures
```

---

## `scenarios/`

JSON files containing recorded road profiles (arc-length vs wheel height) from real-world test drives. Load via:

```python
gen = RoadGenerator.from_scenario_file("config/scenarios/sb1_v7_full.json")
```

or with `just eval`:
```bash
just eval models/.../PPO_final.zip --road recorded \
    --vecnorm-path models/.../vecnormalize.pkl
```

| File | Description | v_ref | Arc length |
|------|-------------|-------|-----------|
| `sb1_v7_full.json` | Full speed-bump run, Nardo FL wheel | 7 m/s | 83 m |
| `sb1_v5_full.json` | Same run replayed at reduced speed | 5 m/s | 83 m |
| `sb1_v3_creep.json` | Same run at creep speed | 3 m/s | 83 m |
| `sb1_v7_approach.json` | Pre-bump flat approach only | 7 m/s | 30 m |
| `exp_v10_early.json` | Early highway, smooth | 10 m/s | 200 m |
| `exp_v12_rough.json` | Roughest 100 m segment | 12 m/s | 100 m |
| `exp_v9_rough2.json` | Second-roughest segment | 9 m/s | 100 m |
| `exp_v14_fast.json` | High-speed rough section | 14 m/s | 100 m |
| `exp_v6_slow.json` | Early highway, low speed | 6 m/s | 150 m |
| `exp_v10_long.json` | Long mixed-rough section | 10 m/s | 300 m |

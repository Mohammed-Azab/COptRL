# Configuration Reference

All YAML files in this directory are loaded at runtime. CLI flags always override YAML values. Changes take effect immediately — no reinstall needed.

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
  n_steps            categorical [1024, 2048, 4096]
  batch_size         categorical [64, 128, 256]
  n_epochs           int        [5, 20]
  gamma              float      [0.97, 0.999]
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

---

## `gym_env/env_params.yaml`

Physics constants and episode settings. Changing these values requires reinstalling the gym package (`just build-gym-env`).

```
m_B, m_W            sprung / unsprung masses [kg]
c_T                 tyre damping [N·s/m]
k_T                 tyre radial stiffness [N/m]
k_S                 suspension spring stiffness [N/m]
D, A                damper characteristic (D = slope, A = compression/rebound ratio)
v_d, v_z            velocity thresholds for high-speed damper slopes [m/s]
f1_cmp / f2_cmp / f1_rbd / f2_rbd   bumpstop progression factors
dz_cmp / dz_rbd     bumpstop clearance limits [m]
F_ks_nlin_max       bumpstop force cap [N]
DT                  control step [s]  (default 0.02 = 50 Hz)
DT_SIM              ODE sub-step [s]  (default 0.001 = 1 kHz)
EPISODE_STEPS       maximum steps per episode
TRUNC_TRAVEL        suspension travel truncation limit [m]
TRUNC_ZS            body displacement truncation limit [m]
OBS_HIGH            clip bounds for [ζ, ζ̇] in the observation
```

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

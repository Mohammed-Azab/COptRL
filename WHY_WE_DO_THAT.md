# Why We Do That

Non-obvious design decisions that are easy to question but important to keep.
Each entry explains what we do, why we do it, and what breaks if you change it.

---

## Speed: km/h in config, m/s in physics

**What we do:**
Config files (`reward_params.yaml`, `curriculum_params.yaml`, `road_params.yaml`) store all speeds in km/h.
The loaders (`reward_params.py`, `road_params.py`, `curriculum.py`) divide by 3.6 at the YAML→code boundary.
Everything inside the codebase — ODE, rewards, SB3, VecNormalize — operates in m/s.

**Why:**
The ODE is Newton's second law in SI units:

```
m_B * z̈_B = F_spring + F_damper          [kg * m/s² = N]
k_T * (ζ − z_W)                           [N/m * m   = N]
c_T * (ζ̇ − ż_W)                          [N·s/m * m/s = N]
```

Every physical parameter is in SI: `k_S = 27,922 N/m`, `k_T = 262,200 N/m`, `c_T = 500 N·s/m`.
Feed km/h into those equations and forces are wrong by a factor of 3.6, making the suspension dynamics incorrect.
The Numba RK4 integrator, spring/damper kernels, and SB3's running statistics all assume consistent units throughout.

km/h in the config is for human readability — a speedometer shows km/h.
The division by 3.6 at the loader is the single conversion point. Nothing else should convert.

**What breaks if you change it:**
- Feeding km/h speeds into the ODE → forces wrong by 3.6×, suspension unrealistic
- Converting inside the env instead of the loader → some callers convert, others don't, silent bugs
- Adding a second conversion point → double-conversion, speeds off by 3.6² = 12.96×

---

## r_tracking has no dead band — it always penalises distance from v_max

**What we do:**
```python
def r_speed_band(v, v_min, v_upper):
    if v < v_min:
        return -1.0 - ((v_min - v) / v_min) ** 2
    return -((v_upper - v) / v_upper) ** 2   # zero only at v = v_upper
```

**Why:**
A dead band `return 0.0 for v in [v_min, v_max]` means every speed in that range is equally
free. The agent exploits the cheapest option — the lowest speed that avoids the stopping
penalty, which was v_min + ε = ~9 km/h (exp_7). At 9 km/h, the agent never reaches the
bump and collects the terminal bonus, giving +100 episode return.

Following Mandl (2021) Eq. 4.21b: `J_speed = Qv × (v_ref − v)²` — the cost is zero only
at exactly v_ref. The dead band was added with good intentions (allow slowing near bumps)
but the comfort terms (r_heave, r_accel) already create that effect. The two terms together
produce the optimal speed automatically: high tracking cost at low speed balances low heave
cost, and the agent finds the cross-over — which is the desired bump-crossing speed.

See `TRIAL_ERROR.md` Issue 3.

**What breaks if you change it:**
Re-introducing a dead band (any range where r_tracking = 0) re-enables the creep exploit.
The agent will find the cheapest speed in the zero-cost zone and stay there.

---

## r_tracking is not velocity-scaled

**What we do:**
The step reward is:
```python
total = (v / v_max) * comfort_core + tracking_penalty
```
The comfort terms (heave, wheel, accel, jerk, smooth) are all multiplied by `v/v_max`.
`r_tracking` is added separately at full strength.

**Why:**
Velocity scaling was introduced to prevent a stop-and-wait degenerate policy: if all rewards are scaled by v/v_max, then at v≈0 all penalties vanish and the agent has no incentive to move.
But if tracking is also scaled, the speed penalty also vanishes at v≈0, which is precisely the situation where we most need it to fire.
Keeping tracking outside the scale means the agent always pays the full stopping penalty regardless of speed.

See `TRIAL_ERROR.md` Issue 1 for the full diagnosis of the degenerate policy this fixed.

**What breaks if you change it:**
Moving `r_tracking` back inside the velocity-scaled block re-enables the stop-and-wait exploit.
The agent learns to stop before the bump (v≈0 → all penalties ≈ 0 → terminal bonus +100 → high return).

---

## Terminal bonus checks mean speed, not just RMS accel

**What we do:**
```python
def compute_terminal_bonus(rms_accel, mean_speed, cfg):
    if rms_accel < cfg.a_limit and mean_speed >= cfg.v_min:
        return cfg.terminal_bonus
    return cfg.terminal_penalty
```
The bonus only fires if the agent both achieved low body acceleration AND maintained a minimum average speed.

**Why:**
A stopped vehicle has `rms_accel = 0` — perfect comfort by definition, because there is no road excitation.
Without the speed check, a stopped agent always passes the comfort test and collects +100 at every episode end.
The speed gate forces the agent to demonstrate comfort while actually driving over obstacles.

See `TRIAL_ERROR.md` Issue 1.

**What breaks if you change it:**
Removing the speed check → stopped agents collect terminal bonus → stop-and-wait policy re-emerges.

---

## a_B_comfort = 0.5 m/s², not 9.81

**What we do:**
`a_B_comfort = 0.5 m/s²` normalises the body vertical acceleration penalty:
```
r_heave = −(clip(z̈_B, ±1.0) / 0.5)²
```

**Why:**
ISO 2631-1:2016 defines 0.5 m/s² RMS as the onset of "fairly uncomfortable" vertical vibration for seated passengers.
Using 9.81 m/s² (gravitational acceleration) makes the penalty negligible at realistic accelerations:
at `z̈_B = 2.5 m/s²` (typical untrained model), `r_heave = −(2.5/9.81)² = −0.065` — almost zero.
With 0.5 m/s², the same excitation gives `r_heave = −4.0` (clipped), a meaningful signal.

**What breaks if you change it:**
Setting `a_B_comfort` too large → heave penalty negligible → agent ignores vertical comfort, learns only speed tracking.
Setting it too small (e.g. 0.1) → heave penalty dominates everything → agent stops to avoid any excitation.

---

## Quadratic reward terms, not quartic (unlike Mandl 2021)

**What we do:**
All reward terms use squared penalties: `−(x/x_ref)²`.

**Why:**
Mandl (2021) recommends quartic body-acceleration cost (`z̈_B⁴`) because it drives speed toward zero more aggressively near obstacles.
In RL, quartic gradients are extreme when the policy is random early in training — the large negative values cause value function overflow and policy collapse before the agent has learned anything useful.
Quadratic terms provide stable gradients throughout training while still penalising large accelerations.
The relative weighting preserves Mandl's intent: comfort and speed terms are roughly equal importance.

**What breaks if you change it:**
Switching to quartic may cause NaN losses or reward explosions in the first million steps when the policy is still random and body accelerations are large (2–4 m/s²).

---

## VecNormalize is shared between train and eval env (obs only, not reward)

**What we do:**
The eval environment is created with:
```python
eval_venv.obs_rms  = train_venv.obs_rms   # shared reference
eval_venv.training = False                 # does not update stats
eval_venv.norm_reward = False              # reward not normalised
```

**Why:**
The policy was trained on normalised observations using the training env's running mean/variance.
If eval used its own separate normalisation, the policy would receive differently-scaled inputs and produce meaningless actions.
Reward is not normalised during eval because we want to report real reward values, not a normalised proxy.

**What breaks if you change it:**
Giving eval its own `obs_rms` → policy sees different observation scale → poor performance even for a well-trained model.
Normalising eval reward → reported episode returns are meaningless, can't compare across runs.

---

## Curriculum does not apply to eval env

**What we do:**
`CurriculumWrapper` is applied to the training env only. The eval env always uses the default road without curriculum level injection.

**Why:**
Curriculum controls training difficulty so the agent sees an appropriate distribution early in training.
Eval must always measure on the same difficulty (full random road) regardless of training step, so that the eval return is a consistent signal and the `EvalCallback` best-model checkpoint reflects real performance.
If eval also used curriculum, early checkpoints would be evaluated on easy roads and appear better than they are.

**What breaks if you change it:**
Applying curriculum to eval → early checkpoints appear good because they are evaluated on level-0 easy roads → best_model.zip is not actually the best model on real difficulty.

---

## Preview wrapper sits inside VecNormalize, not outside

**What we do:**
Wrapper stack (inside to outside):
```
QuarterCarEnv → PreviewWrapper → CurriculumWrapper → Monitor → DummyVecEnv → VecNormalize
```
PreviewWrapper is applied before vectorisation and normalisation.

**Why:**
VecNormalize tracks a running mean and variance for each observation dimension.
The preview slots (9 features) must be part of the observation space that VecNormalize sees from the start so they get their own normalisation statistics.
If PreviewWrapper were applied after VecNormalize, the preview features would bypass normalisation entirely — the policy would see raw [0,1] preview values alongside normalised base observations, breaking the input scale consistency.

**What breaks if you change it:**
Wrapping after VecNormalize → preview features not normalised → different input scale to the policy → training instability and poor use of preview information.

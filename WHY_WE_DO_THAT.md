# Why We Do That

Non-obvious design decisions that are easy to question but important to keep.
Each entry explains what we do, why we do it, and what breaks if you change it.

---

## Curriculum uses pre-generated scenarios, not on-the-fly random roads

**What we do:**
`scripts/generate_train_scenarios.py` produces 200 YAML files per difficulty level (easy/medium/hard/expert) and writes them to `config/train/scenarios/` before training starts. `CurriculumWrapper` loads them once at init and samples uniformly from the pool on each reset. Fallback random-road parameters (used when files are absent or for the expert-level multi-bump fraction) live in `config/curriculum/curr_multi_bumps.yaml` separately from the main `curriculum_params.yaml`.

**Why:**
On-the-fly random roads (`RoadGenerator.from_random`) are fully unbounded — any catalog bump at any speed in the level's range. This means the agent can see the same (bump, speed) pair in multiple episodes at slightly different values, or never see some extreme combinations at all within a training run. Pre-generation fixes the training distribution: every scenario is distinct, difficulty boundaries are exact (based on `difficulty_score = 0.6 × bump_rank + 0.4 × speed_norm`), and the set is reproducible from the seed. It also lets us inspect and adjust the distribution (run the script again) without changing training code.

Level 3 (expert) still mixes in 25% random multi-bump episodes via `allow_multi_bump: true` so the agent does not overfit to single-bump trajectories at the highest level.

**What breaks if you change it:**
- Deleting `config/train/scenarios/` without removing `scenarios_dir` from the curriculum config → wrapper silently falls back to random generation at all levels, training still works but loses distribution control.
- Changing thresholds in the script after training has started → the new difficulty assignments won't match what the agent has already seen; reload scenarios and restart from level 0.
- Removing `curr_multi_bumps.yaml` → fallback branch in `CurriculumWrapper.reset()` raises `KeyError` on `num_bumps_range`; always keep it alongside `curriculum_params.yaml`.

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

---

## Optuna does not tune seed

**What we do:**
Each Optuna trial receives a fixed seed (`base_seed + trial.number`) for reproducibility.
Seed is not in the search space (`tune_config.yaml`) and Optuna never suggests different seeds
as a hyperparameter.

**Why:**
Seed controls randomness, not learning capacity. Tuning it is selection bias, not optimisation.

A real hyperparameter (learning rate, batch size, gamma) changes *how well the algorithm
learns*. The same learning rate performs consistently better or worse regardless of random draw.

Seed controls *which random outcomes occur* — road geometry on episode 1, weight initialisation,
mini-batch ordering. If trial A with seed=42 happens to get easy roads during eval and trial B
with seed=7 gets hard roads, trial A looks better even with identical hyperparameters.
Optuna would select seed=42 as "best" — but the next training run with that seed will likely
perform average because road randomisation reintroduces variance the seed cannot control.

This is called **selection bias**: you are not finding a better algorithm, you are finding a
lucky draw.

**What we do instead:**
Each trial evaluates over `n_eval_episodes=5` and the objective is the **mean** return across
those episodes. Averaging over multiple episodes reduces seed sensitivity without exploiting it.
This is the statistically correct approach — reduce variance through aggregation, not selection.

**What breaks if you change it:**
Adding seed to the Optuna search space means the study will converge on the luckiest seed, not
the best hyperparameters. Results will not reproduce: retraining with the "winning" seed gives
average performance, not the performance the study measured.

---

## Geometry clamp uses physics formula, not empirical fit

**What we do:**
```python
ZETA_DOT_LIMIT = 7.0   # m/s — matches OBS_HIGH[1] in env_params.yaml
steepest = max(π * H / L for each bump)
v_lim = ZETA_DOT_LIMIT / steepest
self.speed = clip(self.speed, 0, v_lim)
```

**Why:**
The original clamp used a linear fit from two empirical data points (ba_azab runs 11 and 13).
Both points had nearly identical speeds (~22 km/h), so the fitted line was near-horizontal
and clamped every bump — including a gentle 4 cm × 7 m bump — to about 23 km/h.
Result: 56–91% of curriculum episodes were clamped to the same speed regardless of level,
making the curriculum ineffective (all levels produced the same ~20 km/h speed distribution).

The physics-based formula sets the limit so the peak road velocity ζ̇ = v · πH/L stays
within the observation window (7 m/s). For gentle bumps (low H/L ratio) the limit is high
and no clamping occurs. For steep bumps the limit is appropriately low.

**What breaks if you change it:**
Reverting to the empirical fit clamps all bumps to ~20 km/h regardless of geometry.
The agent never sees speeds above ~23 km/h during training and cannot learn meaningful
speed planning — the task reduces to "always drive at 20 km/h".

---

## Curriculum heights are capped at 15 cm (not 30 cm)

**What we do:**
```
Level 0: 4–7 cm   (Nardo reference: 6.4 cm)
Level 1: 4–10 cm
Level 2: 5–12 cm
Level 3: 5–15 cm  (upper end of real sleeping-policeman bumps)
```

**Why:**
Real-world speed bumps are 7–12 cm tall (EU standard) and the Nardo proving ground data
used to calibrate this model has a maximum height of 7.8 cm. 30 cm bumps (the previous
level 3 maximum) do not exist as speed bumps — they are curbs or steps. At 30 cm height
the geometry clamp forces the speed to 13.6 km/h regardless of bump length, removing any
speed planning challenge and making the training distribution unrealistic.

15 cm is the upper end of aggressive real sleeping-policeman bumps and gives the agent
a physically meaningful challenge: slow down from 50–72 km/h to ~30–40 km/h, cross
smoothly, resume speed.

**What breaks if you change it:**
Raising heights above 15–20 cm → geometry clamp forces extreme speed reduction → agent
learns to creep at 14 km/h on all roads → no meaningful speed planning learned.

---

## Road completion is termination, not truncation

**What we do:**
When the agent reaches `max_distance` (end of road past the last bump), the episode ends
as `terminated=True` and the terminal bonus fires based on RMS body accel and mean speed.

**Why:**
Previously `max_distance` caused `truncated=True`, which skips the terminal bonus entirely.
An agent that crossed all bumps quickly and comfortably at step 170 got zero terminal signal.
The agent that crept to step 300 got ±100. This gave a perverse incentive to waste time
rather than cross efficiently.

With `terminated=True` on road completion, both paths (fast crossing and slow crossing)
receive the same terminal evaluation. The agent is rewarded for quality of crossing, not
for how long it took.

**What breaks if you change it:**
Reverting to `truncated=True` on road completion removes the terminal signal for all episodes
where the agent crosses the road efficiently. PPO's advantage estimation is cut off and the
agent never learns that fast+comfortable crossing is the goal.

---

## TRUNC_TRAVEL = 0.20 m, not 0.15 m

**What we do:**
Safety truncation for suspension travel fires at 0.20 m dynamic travel (z_W − z_B from
static equilibrium).

**Why:**
Calibrated from 100 worst-case episodes (6 × 15 cm × 1 m bumps, full throttle):
maximum observed travel was 0.149 m. Setting the limit to 0.15 m would fire spuriously
on unlucky integration steps (0.001 m margin is insufficient). 0.20 m gives 34% margin
while still catching genuine numerical blow-ups (ODE divergence), which would produce
values far above 0.20 m.

The original 0.60 m limit never fired — it was unreachable without numerical divergence
and provided no meaningful safety net. 0.20 m is tight enough to catch real problems and
loose enough to never interfere with normal training.

**What breaks if you change it:**
Setting below 0.15 m → spurious safety truncations during fast aggressive crossings →
episodes cut short, no terminal bonus, noisy training signal.
Setting to 0.60 m → reverts to dead code that never fires.

---

## r_jerk and r_action_smooth are not velocity-scaled

**What we do:**
```python
jerk_smooth = w_jerk * r_jerk + w_smooth * r_smooth   # unscaled
core = w_heave*r_heave + w_wheel*r_wheel + w_accel*r_accel  # scaled by v/v_max
total = (v/v_max) * core + tracking_penalty + jerk_smooth
```

**Why:**
Jerk and action_smooth measure self-induced longitudinal oscillation — how rapidly the agent
changes its acceleration command. The agent fully controls this at any speed.

At 13.6 km/h (scale=0.189) the velocity-scaled version discounted jerk to 18.9% of its cost
at full speed. The agent in exp_10 learned to drive slowly and oscillate wildly: −20 jerk
penalty per episode instead of −108. It hovered near the comfort threshold and collected
terminal bonuses while producing useless motion.

Heave and wheel acceleration physically depend on road excitation × speed — there is a genuine
physical reason to scale them down at low speed (smaller road disturbances at lower velocity).
Jerk does not. An agent oscillating between braking and accelerating at 13.6 km/h is exactly
as uncomfortable as at 72 km/h.

See `TRIAL_ERROR.md` Issue 5.

**What breaks if you change it:**
Moving jerk/smooth back inside the velocity-scaled block re-enables the oscillation exploit.
The agent drives slowly, oscillates freely, and collects terminal bonuses without ever crossing
bumps smoothly.

---

## step_bonus: constant per-step reward shift

**What we do:**
```python
total = scale * core + tracking_penalty + jerk_smooth_penalty + progress_reward + cfg.step_bonus
```
`step_bonus = 0.5` is added to every step unconditionally, shifting all episode returns by approximately +150 (300 steps × 0.5).

**Why:**
Without the bonus, the per-step reward range is [−9.69, +0.20] — heavily skewed negative.
Typical returns before the bonus:
- Human driver (random road): mean −123, max +35
- MPC baseline: mean −15.7
- RL agent during tuning: best trial +50

The overwhelming negativity creates two problems:
1. **Misleading signal** — the agent sees mostly negative rewards even when it is doing well, making it hard to distinguish "slightly bad" from "very bad". A step with no bumps and good speed tracking returns −0.05 (tiny negative), indistinguishable from a mildly bad step.
2. **Curriculum calibration** — positive thresholds are easier to reason about and calibrate against observed baselines.

With step_bonus=0.5:
- A step with no bumps, at v_ref, returns +0.65 (clearly positive)
- A bump-crossing step with comfort issues returns −0.3 to −0.9 (clearly negative)
- Human driver mean: −123 → +19.8 (barely positive = "doing OK but not great")
- The threshold for curriculum advancement maps directly to a meaningful performance level

The value 0.5 was chosen so the human driver's mean sits just above zero — neutral performance is near zero, clearly good is clearly positive, clearly bad is clearly negative.

**What breaks if you change it:**
- `step_bonus=0`: returns to the pre-shift scale; curriculum thresholds (+190/+175/+155) become too high.
- `step_bonus >> 1`: episode returns are all large and positive; the relative difference between good and bad policies shrinks (signal-to-noise decreases).
- If you change `step_bonus`, update curriculum thresholds by `Δbonus × mean_episode_length`.

---

## r_progress: positive reward for forward movement

**What we do:**
```python
r_progress = v / v_max   # in [0, 1], unscaled, always on
total = scale * core + tracking_penalty + jerk_smooth_penalty + w_progress * r_progress
```

**Why:**
Before this, every reward term was non-positive. The agent had:
- Penalties for going slow (r_tracking)
- Penalties for bad comfort (r_heave, r_accel, ...)
- A terminal bonus/penalty at the end

But **no positive signal that directly rewards forward progress**. The only incentive to move was avoiding the tracking penalty. With enough other penalties active (heave, jerk), the agent could balance near a low-but-non-zero speed and the gradient toward faster movement was weak.

`r_progress = v/v_max` is a direct, always-positive encouragement: the faster you move, the more you earn each step. At v=0 it contributes 0; at v=v_max it contributes w_progress × 1 = +0.2/step = +60 per episode on top of everything else.

Combined with r_tracking (negative for being below v_max), the speed signal has both:
- A pull (positive progress reward → go faster)
- A push (tracking penalty → get away from zero)

This widens the gap between the creep-and-wait exploit (+39) and genuine good crossing (+64) from 3 points to 25 points, making learning more reliable.

The new episode maximum is +160 (full speed, flat road, zero heave), up from +100.

**What breaks if you change it:**
Removing r_progress narrows the reward gap between creeping and crossing.
Setting w_progress too high (> 0.5) risks the agent going full speed regardless of bumps
because the progress reward outweighs heave penalties.

---

## Preview observation uses t2r and freq, not dist and width

**What we do:**
The preview wrapper encodes each upcoming bump as `[t2r, height, freq]` rather than `[dist, height, width]`.

```python
t2r  = dist_m / v_safe          # time-to-reach in seconds, normalised by T_MAX = preview_distance / v_min
freq = v_safe / peak_w_m        # crossing frequency in Hz, normalised by _FREQ_MAX = v_max / L_narrowest
```

**Why — this is not cheating:**
`dist` and `width` contain the same information, but in a form that requires the neural net to learn to divide by `v` before the signal is useful.
`t2r` is *Time-to-Contact (TTC)* — the standard automotive safety metric. A driver does not think "that bump is 30 m away"; they think "I have 1.5 s." A policy that receives raw distance misses urgency entirely: 30 m at 20 m/s (1.5 s) and 30 m at 5 m/s (6 s) look identical but demand opposite actions.
`freq = v / L` is the excitation frequency the suspension will see on crossing. At resonance (body ~1.3 Hz, wheel ~11 Hz) even a small bump produces large accelerations. Providing this directly means the policy doesn't need to approximate a division.

Both features are derived from quantities the real sensor already provides (distance, width, speed). A real control engineer would compute both in one line. Providing them pre-computed removes a hard nonlinear approximation problem and replaces it with a nearly-linear relationship — this is observation engineering, not privilege.

**What breaks if you revert:**
Replacing t2r with raw dist gives identical information but makes the urgency–speed interaction implicit. The policy must then learn to divide, which takes many more samples and may never fully converge. At high speed the agent will routinely under-react to close bumps; at low speed it will over-react to distant ones.

---

## How the agent uses t2r to brake before bumps (not during)

**The problem without preview**

Without any lookahead, the agent only observes its current state (speed, suspension deflection).
By the time `zeta` and `zeta_dot` rise — which is the first physical signal that a bump exists —
the vehicle is already on top of it. Braking at that point only worsens the crossing: abrupt
deceleration mid-bump increases jerk and body accel instead of reducing them.
Reactive control cannot produce a smooth crossing; anticipatory control can.

**What t2r enables**

Every step, `PreviewWrapper` appends `[t2r, height, freq]` for each upcoming bump.
`t2r = dist / v_current` — the seconds until impact at current speed, normalised by `T_MAX`.
As the agent drives toward a bump, `t2r` counts down from ~1 to 0:

```
t2r ≈ 0.9   far away     →  maintain speed, collect progress reward
t2r ≈ 0.4   approaching  →  should be decelerating now
t2r ≈ 0.05  close        →  should already be at crossing speed
t2r = 0     on bump      →  cross at whatever speed you arrived at
```

The agent cannot change what happens at `t2r = 0`. The reward at that step is determined
by what it did at `t2r = 0.4`. This temporal credit assignment is the whole challenge.

**How the policy learns it**

During training the agent tries many actions and observes the resulting rewards.
It learns — through repeated episodes — a pattern:

> "When t2r was ~0.5 and height was large, steps where I had already been decelerating
> gave low heave penalty on crossing. Steps where I hadn't started decelerating yet
> gave large heave penalty. Therefore: at t2r ≈ 0.5 with large height, decelerate."

The policy network learns a mapping `(t2r, height, freq, v, ...) → action`.
It does not execute a formula; it learns a nonlinear function from observation to action
that approximates what a formula would prescribe.

**Why t2r (not dist) makes this easier to learn**

`t2r = dist / v_current` accounts for speed automatically. At 50 km/h you need to start
braking 30 m out (≈ 2.2 s). At 20 km/h you can wait until 12 m (≈ 2.2 s). Both situations
have `t2r ≈ 0.4` — the same observation triggers the same action. With raw distance, the
policy would need to separately learn the speed–distance relationship before it could act
correctly, doubling the approximation problem. t2r collapses both into one number.

**What breaks if the preview is removed**

Without preview the policy sees only the current bump height (zeta, zeta_dot).
Those rise from zero only when the vehicle is on the bump. No information arrives early
enough to support anticipatory braking. The agent can only react — it learns to slow
down reactively during bumps (producing high jerk) rather than proactively before them.
In practice this means consistently worse heave and accel scores even with the same
reward function.

---

## Issue 13 — MPC one-sided speed tracking (and why it doesn't decelerate)

**What we observed:**
MPC baseline scores ~-286 vs RL ~-25 even though the MPC has full road preview.
The agent drives into bumps at full speed rather than braking before them.

**Root cause:**
`src/baseline/mpc/ocp.py` computes the speed-tracking residual as:

```python
speed_err = ca.fmax(0.0, v_ref - v) / v_max_v   # one-sided
yref[2] = v_ref / v_max_v
```

`ca.fmax(0.0, ...)` clamps to zero when `v > v_ref`.
When the vehicle is approaching a bump at v > v_ref, the speed error is **zero** — no cost penalty — so the solver has no incentive to decelerate. The only brake signal would come from the heave/acceleration terms, but those only activate *during* the bump, too late for any useful deceleration.

**Fix:**
Replace one-sided clamping with a symmetric (signed) residual, and set the reference to zero:

```python
speed_err = (v - v_ref) / v_max_v   # two-sided: negative = too slow, positive = too fast
yref[2] = 0.0                         # already 0 in some versions — make explicit
```

With `yref[2] = 0.0` and a quadratic cost `Q * speed_err^2`, the cost is symmetric:
overspeed and underspeed are equally penalised, so the solver decelerates before bumps and
re-accelerates after crossing them.

**Why this was wrong in the first place:**
The original code was written to match the RL reward convention where being *above* v_ref is fine
(there is a `ca.fmax` in the RL reward too). But the RL reward is sparse and unidirectional by
design — the *speed tracking* component only penalises being too slow. MPC uses a quadratic cost
which should be **symmetric around the reference**. Mixing the two conventions caused the MPC to
have a dead zone above v_ref where no speed cost was paid.

**What breaks if you change it back:**
Restoring `ca.fmax(0.0, v_ref - v)` removes the deceleration incentive. The MPC will
re-enter bumps at full speed and produce large heave, exactly as observed.

**Benchmark context (u=0 baseline):**
A constant zero-throttle policy scores ~-359, MPC with the one-sided bug scores ~-286 (~70 better).
MPC with two-sided tracking is expected to score closer to the RL agent (~-25 to -50) because it
can actually plan a brake-coast-accelerate profile over the bump preview horizon.

---

## Mandl 2024 improvements: v_ref in observation and absolute speed tracking

**What changed (2026-06-10):**
1. `v_ref / v_max` added as observation feature (index 6 in the base obs).
2. `r_speed_band` formula above `v_min` changed from one-sided quadratic (`r = 0` in comfort zone, `r < 0` above `v_limit`) to Mandl-style absolute tracking: `r = −|v − v_ref| / v_ref`.

**Why the old formula was suboptimal:**
The original `r_speed_band` had a dead zone between `v_min` and `v_limit` where the reward was exactly 0.  The only incentive to drive fast was `r_progress = v/v_max`. Near bumps, `v_ref` dropped (via `_compute_v_ref`), but the reward gave no gradient toward that lower target — the agent had to discover the optimal braking speed purely through the comfort and bump-crossing terms.

**What the absolute formula adds:**
`r = −|v − v_ref| / v_ref` is zero only at `v = v_ref` and increases linearly as the agent deviates in either direction. This means:
- On flat road (v_ref = v_max): agent is penalised for driving below v_max, same directional pressure as before but with a gradient throughout [v_min, v_max].
- Near a bump (v_ref < v_max from `_compute_v_ref`): agent is penalised for driving faster *or* slower than v_ref. The tracking term now directly encodes "slow to this specific target speed before the bump."

**Why v_ref in observation:**
Without v_ref in the observation, the policy must infer the current target speed from the preview features (t2r, height, freq). Adding `v_ref / v_max` gives the policy an explicit, pre-computed signal of the recommended approach speed, making the speed-tracking task much easier to learn.

**What we kept from COptRL:**
- The cliff penalty below v_min is unchanged — stopping is still heavily penalised.
- All other reward terms (heave, wheel, accel, jerk, smooth, progress, bump cross, terminal) are unchanged.
- The velocity scaling on comfort terms is unchanged.

**Source:** Mandl (2024) "Speed Control in the Presence of Road Obstacles: A Comparison of RL and MPC", reward function Eq. (1a), Q_v term.

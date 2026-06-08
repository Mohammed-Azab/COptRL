# COptRL — Trial & Error Log

Chronological record of training failures, root causes, and fixes. Each entry documents
what went wrong, why, what was tried, and the exact code change that resolved it.

---

## Template

```
## Issue N — Short Description

**Date:** YYYY-MM-DD
**Exp:** exp_N on road=X, timesteps=Y

### Symptom
What the output looked like (metrics, plots, eval numbers).

### Root Cause
The actual reason this happened.

### What We Tried
Things that didn't work and why.

### Fix
The exact code change and why it works.

### Verification
Numbers that confirm the fix worked.

### Lesson
One-line takeaway.
```

---

## Issue 1 — Stop-and-Wait Degenerate Policy

**Date:** 2026-06-07
**Exp:** exp_5, speed_bump, 1M steps, no curriculum

### Symptom

Eval output after 1M training steps:

```
Mean speed          +0.590 m/s        ← agent barely moving
Speed tracking RMSE +19.461 m/s       ← massive deviation from v_ref=20
RMS body accel      +0.000 m/s²       ← perfect comfort (never hit the bump)
Comfort score       +1.000            ← perfect (trivially, by stopping)
Episode return      +86.955           ← high return despite doing nothing useful
r_tracking          -247.17 total
r_heave             +0.00 total       ← never hit a bump
```

The agent learned to brake to a near-stop before the bump, wait for the episode to end
(300 steps × 0.02 s = 6 s), collect the terminal bonus, and never cross the obstacle.

### Root Cause

**Two compounding design flaws:**

**Flaw 1 — Velocity scaling also scaled the tracking penalty.**

The step reward was structured as:

```python
core = w_heave*r_heave + w_wheel*r_wheel + w_tracking*r_tracking + w_accel*r_accel + ...
total = (v / v_max) * core
```

At `v = 0.59 m/s` (agent stopped), `v/v_max = 0.0295`. Every term — including the tracking
penalty — was multiplied by 0.03. The per-step tracking penalty shrunk from `0.8 × (−1.0) = −0.8`
to `0.0295 × 0.8 × (−1.0) = −0.024`. Over 300 steps that is only −7 total tracking penalty.

The velocity scaling was designed to discourage stopping (low-speed → small comfort rewards).
But it made the speed penalty nearly vanish too — the exact opposite of what we needed.

**Flaw 2 — Terminal bonus did not check mean speed.**

```python
def compute_terminal_bonus(rms_accel, cfg):
    if rms_accel < cfg.a_limit:   # only checks body accel, not speed
        return cfg.terminal_bonus  # +100
    return cfg.terminal_penalty
```

A stopped agent has `rms_accel = 0` (no excitation → no body movement → perfect comfort).
So `0 < 1.0 m/s²` → terminal bonus +100 was always awarded to any agent that stopped.

**Combined effect:** stop before bump → tracking penalty ≈ −7 → terminal bonus +100 → episode
return ≈ +87. This was the highest achievable return the agent could find.

### What We Tried

Before implementing the fix, we considered:

1. **Increase w_tracking** — Would make tracking penalty bigger, but since it's still scaled
   by v≈0, at best −7 × k for any multiplier k. Doesn't fix the root cause.

2. **Remove velocity scaling entirely** — Would prevent the comfort terms from being zero at
   low speed, which could cause instability on flat road (accel penalty fires even at rest).
   Also, Mandl's original insight (comfort terms should be less penalised at low speed) is
   correct — we just shouldn't apply it to the tracking term.

3. **Increase terminal penalty** — Doesn't help since the terminal bonus is what fires here
   (rms=0 passes the comfort check).

### Fix

Two changes, in `reward/reward.py` and `envs/quarter_car_env.py`:

**1. Pull `r_tracking` outside the velocity-scaled block** (`reward/reward.py`):

```python
# Before
core = ... + w_tracking * r_tracking + ...
total = (v/v_max) * core

# After
tracking_penalty = w_tracking * r_tracking  # full strength always
comfort_core = w_heave*r_heave + w_wheel*r_wheel + w_accel*r_accel + ...
total = (v/v_max) * comfort_core + tracking_penalty
```

Rationale: velocity scaling is meant to say "comfort penalties matter less at low speed
because the road excitation is smaller". Speed deviation is not a comfort penalty — it
is the anti-stopping gradient. It must fire at full strength regardless of speed.

**2. Add a mean-speed gate to the terminal bonus** (`reward/reward.py`):

```python
# Before
def compute_terminal_bonus(rms_accel, cfg):
    if rms_accel < cfg.a_limit:
        return cfg.terminal_bonus

# After
def compute_terminal_bonus(rms_accel, mean_speed, cfg):
    if rms_accel < cfg.a_limit and mean_speed >= cfg.v_min:
        return cfg.terminal_bonus
```

Called in env as:
```python
mean_speed = self._s_pos / max(self._t, 1e-9)
reward += compute_terminal_bonus(rms, mean_speed, cfg)
```

Rationale: the terminal bonus should only reward an agent that achieved good comfort
**while also moving**. A stopped agent with RMS=0 doesn't deserve a comfort reward.

### Verification

Simulated three policies with the fixed reward:

```
Stop-and-wait (v=0.59, z̈_B=0):   −219.3   ← now correctly worst
Good policy   (v=10,   z̈_B=0.4): +23.2    ← best: slowed smoothly for bump
Full speed    (v=20,   z̈_B=2.5): −1060.0  ← worst comfort
```

Stop-and-wait is now −219 vs. +23 for good behaviour — a 240-point gap creating a clear
training signal toward the desired behaviour.

### Lesson

**Velocity scaling and speed-tracking penalty are incompatible in the same reward block.**
Any mechanism intended to penalise low speed must be applied *outside* the velocity-scaling
factor, otherwise the scaling itself removes the gradient signal it was meant to preserve.

**Terminal bonuses must check the same objectives as per-step rewards.** A comfort bonus
awarded to an agent that stopped is not measuring comfort — it is rewarding inactivity.

---

## Issue 2 — Training with or without Curriculum?

**Date:** 2026-06-07
**Context:** After fixing Issue 1, deciding whether to enable curriculum for future runs.

### Observation

Exp_5 (no curriculum) trained for 1M steps and still found a degenerate policy. Exp_4 (also
no curriculum) achieved `mean_reward = 59.9` with `min_reward = −280`, suggesting the agent
was exploring diverse strategies but not reliably converging.

### Analysis

**Without curriculum:**
- At step 0, the agent immediately faces random bumps of height 0.05–0.25 m at speeds up to 20 m/s
- With random actions, bumps at 20 m/s produce body accelerations of ~2–3 m/s², leading to
  large negative rewards from the start
- The gradient signal is noisy — the agent receives wildly different returns depending on which
  random road was sampled
- The agent may learn the degenerate stop-and-wait policy as a safe local optimum

**With curriculum (3 levels):**

| Level | Steps | Bumps | Height | Speed |
|-------|-------|-------|--------|-------|
| 0 | 0–200k | 1–2 | 5–10 cm | 4–10 m/s |
| 1 | 200–500k | 1–3 | 5–15 cm | 4–15 m/s |
| 2 | 500k+ | 1–5 | 5–25 cm | 4–20 m/s |

Level 0 provides a stable learning signal: small bumps at moderate speed are forgiving.
The agent can learn the basic approach (slow down a little, cross, resume) without encountering
the extreme truncations and huge negative returns of full-difficulty scenarios.

### Recommendation

**Always use `--curriculum` (`just train speed_bump --c`).** Reasons:

1. Prevents the agent from being overwhelmed by hard scenarios before it has learned basic speed control
2. The gradient signal at Level 0 is more consistent, leading to faster convergence on the
   fundamental skill (crossing without stopping)
3. Level 2 difficulty (5 bumps, 0–20 m/s) is eventually reached, so final policy quality is
   the same as without curriculum — it just takes less total timesteps to get there

**Random road per episode is essential** (default, not an issue):

The `from_random()` call at each `reset()` ensures the agent never memorises a single bump
layout. Without this, a policy trained on a fixed road fails on any other road. Keep
`random_road_on_reset = True`.

### Lesson

Enable curriculum. It costs nothing (no code change, just `--c` flag) and consistently
produces more stable early training. The agent learns the skill incrementally rather than
having to solve all difficulty levels simultaneously.

---

## Issue 3 — Creep-and-Wait Exploit (Dead Band in r_tracking)

**Date:** 2026-06-08
**Exp:** exp_7, speed_bump, 1M steps, curriculum, seed=69

### Symptom

Eval output after 1M steps with curriculum:

```
Mean speed          8.963 km/h       ← creeping just above v_min
Speed tracking RMSE 63.35 km/h       ← massive deviation from v_ref=72 km/h
RMS body accel      0.000 m/s²       ← never hits the bump
Comfort score       1.000
Episode return      +45 to +65       ← positive returns (exploit still paying off)
r_tracking          -52.68 total     ← small penalty
r_heave             +0.00 total      ← never crossed an obstacle
```

The agent learned to creep at ~9 km/h — just above v_min (7.2 km/h) — for the entire episode
without ever reaching the bump. This is a more subtle variant of the stop-and-wait exploit
from Issue 1.

### Root Cause

`r_speed_band` returned zero for all speeds in the range `[v_min, v_max]`:

```python
def r_speed_band(v, v_min, v_upper):
    if v < v_min:
        return -((v_min - v) / v_min) ** 2
    if v > v_upper:
        return -((v - v_upper) / v_upper) ** 2
    return 0.0   # ← any speed in [7.2, 72] km/h costs nothing
```

At v = 8.9 km/h (above v_min = 7.2 km/h), `r_tracking = 0`.
The velocity scaling factor at that speed: `8.9/72 = 0.124` — comfort terms are 88% discounted.
The terminal bonus mean-speed gate (`v >= v_min = 7.2 km/h`) is satisfied by v=8.9 km/h.

So the agent could collect terminal +100 with zero tracking penalty and near-zero comfort
penalties, giving an episode return of +100.0 — the theoretical maximum.

```
Pre-fix episode returns (simulated):
  creep at 8.9 km/h, rms=0:    +100.0   ← free exploit
  good  at 54  km/h, rms=0.5:   −80.0   ← penalised by heave
  full  at 72  km/h, rms=2.5:  −1060.0
```

Issue 1 fixed stopping at v≈0 by unscaling r_tracking. But the dead band `[v_min, v_max]`
remained, meaning any speed between 7.2 and 72 km/h was still free. The agent simply
found the next available minimum: v_min + ε.

### What We Tried

- Raising v_min — reduces the dead band but doesn't eliminate it; the agent would just creep
  at the new minimum
- Raising the mean-speed gate in the terminal bonus — delays the exploit but doesn't remove it

### Fix

Remove the dead band entirely. Follow Mandl (2021) Eq. 4.21b exactly:
`J_speed = Qv × (v_ref − v)²` — always nonzero unless v = v_ref.

```python
# Before
def r_speed_band(v, v_min, v_upper):
    if v < v_min:
        return -((v_min - v) / v_min) ** 2
    if v > v_upper:
        return -((v - v_upper) / v_upper) ** 2
    return 0.0   # dead band

# After
def r_speed_band(v, v_min, v_upper):
    if v < v_min:
        return -1.0 - ((v_min - v) / v_min) ** 2   # extra penalty below minimum
    return -((v_upper - v) / v_upper) ** 2           # always penalise distance from v_max
```

At v=8.9 km/h: `r_tracking = -((72-8.9)/72)² = -0.768` (was 0).
At v=54 km/h: `r_tracking = -((72-54)/72)² = -0.0625` (small, not zero).
At v=72 km/h: `r_tracking = 0` (only at exactly v_max).

This creates a continuous gradient toward v_max. The agent can still justify slowing
for a bump because the heave reduction outweighs the tracking penalty increase — but
it can no longer exploit a free zone.

### Verification

Post-fix simulated returns:

```
creep 8.9 km/h, rms=0 (old exploit): −84.3   ← now heavily penalised
moderate 36 km/h, rms=0.3:            −3.2   ← best: balanced trade-off
good 54 km/h, rms=0.5:               −95.0   ← (constant rms over 300 steps, pessimistic)
```

The optimal speed (~36 km/h in the constant-rms simulation) corresponds to where the
tracking penalty and comfort penalty balance — exactly the trapezoidal velocity profile
Mandl describes as the desired human-like behaviour.

### Lesson

**A dead band anywhere in the reward creates a free zone the agent will exploit.**
Any `return 0.0` in a reward term that the agent can reliably reach is a potential local
optimum that produces useless behaviour. Follow Mandl's formulation: tracking is a
continuous squared deviation from v_ref, not a band. The dead band was added with good
intentions (allow slowing for bumps) but the comfort terms already create that effect
naturally — the dead band is redundant and harmful.

**The tracking term and comfort terms together create the optimal speed automatically.**
No dead band is needed. At low speed: tracking penalty is large, comfort penalty is small.
At high speed: tracking penalty is small, comfort penalty is large. The agent finds the
speed where the two balance — which is the desired bump-crossing speed.

---

## Issue 4 — All-Negative Rewards After Dead-Band Removal (w_tracking Miscalibration)

**Date:** 2026-06-08
**Exp:** exp_8, speed_bump, 1M steps, curriculum, seed=69

### Symptom

Training summary after exp_8 (first run after Issue 3 fix):

```
episodes     : 3587
mean_reward  : -273.971    ← all negative
max_reward   :  -53.942    ← even the best episode is negative
min_reward   : -534.028
mean_ep_len  :  279.1      ← truncations happening (should be 300)
```

Every single episode has negative return. The best model (step 620k) is used only because
it is the least bad, not because it learned anything useful.

Eval of exp_8's best model still showed creep behaviour at 8.9 km/h, suggesting the agent
gave up trying to cross bumps and just sought the lowest-penalty path.

### Root Cause

Issue 3 removed the dead band from `r_tracking` so the formula always fires:
```python
return -((v_upper - v) / v_upper) ** 2   # zero only at v = v_upper = 72 km/h
```

But `w_tracking = 0.8` was calibrated for the old formulation where tracking = 0 for any
speed inside [v_min, v_max]. With the dead band, w_tracking was essentially a stopping penalty
— it only mattered below v_min = 7.2 km/h.

Without the dead band, w_tracking is a constant per-step cost for any speed below v_max.
At curriculum level 0 (max episode speed = 36 km/h), the agent incurs:

```
tracking penalty = 300 steps × w_tracking × r_tracking
                 = 300 × 0.8 × −((72−36)/72)²
                 = 300 × 0.8 × −0.25
                 = −60 per episode — unavoidable at level 0
```

This creates a −60 baseline tracking cost every episode that the agent cannot reduce
regardless of what it does (curriculum limits speed to 36 km/h max at level 0, so
r_tracking cannot be zero).

Combined effects:
- Good episode (rms=0.3, terminal +100): −60 tracking − 43 heave + 100 = −3 (barely positive)
- Bad episode (rms > 1, terminal −100):  −60 tracking − X heave − 100 = −200 to −300
- Truncated episode (no terminal):       −60 tracking − X heave + 0   = −100 to −200

The agent found that most realistic bump crossings give rms > 1 m/s² (especially at the
higher speed end of level 0), so the terminal penalty fires frequently → mean around −274.

### Fix

Reduce `w_tracking` from 0.8 to 0.3. The weight must be recalibrated whenever the
structural form of r_tracking changes — the dead-band removal made tracking always-on,
so the weight needed to be reduced proportionally.

With w_tracking = 0.3 at level 0 (36 km/h):
```
tracking penalty = 300 × 0.3 × −0.25 = −22.5 per episode
```

Level 0 episode returns (fixed):
```
36 km/h, rms=0.3 (good comfort):    +34.3   ← clearly positive, learnable
36 km/h, rms=1.5 (bad comfort):    −602.5   ← strong signal to slow for bumps
9  km/h, creep:                     +31.1   ← still positive but < good policy
```

The good policy (+34) is now strictly better than creeping (+31) and the gradient is clear.

### Lesson

**Reward weights are coupled to reward structure, not independent knobs.**
When the structural form of a reward term changes (dead band → always-on), all weights
touching that term must be recalibrated. Keeping w_tracking = 0.8 after removing the
dead band was equivalent to multiplying the old stop-penalty by 11× for every in-band
speed (previously zero, now 0.8 × 0.25 = 0.2/step).

A simple sanity check after any reward change: simulate a "good" policy and verify
the episode return is positive. If the best achievable behaviour gives negative return,
the weights are wrong.

---

## Issue 5 — Low-Speed Oscillation Exploit (Jerk/Smooth Velocity-Scaled)

**Date:** 2026-06-08
**Exp:** exp_10, speed_bump, 1M steps, curriculum, seed=69, tuned hyperparameters

### Symptom

Eval of best model (step 160k):

```
Mean speed           13.584 km/h      ← very slow (target 54–72 km/h)
Speed tracking RMSE  58.48 km/h       ← massive deviation from v_ref
RMS body accel       0.821 m/s²       ← borderline (just under a_limit=1.0)
Comfort score        0.179            ← very low despite borderline RMS
r_jerk               -542.06 total    ← ENORMOUS — biggest single penalty
r_accel              -255.09 total
r_tracking           -197.92 total
Action smoothness RMS  0.553          ← high (confirms wild oscillation)
```

The agent drives at 13.6 km/h with constant rapid acceleration changes (high jerk, high
action_smooth penalties), hovering just under the RMS comfort threshold to sometimes collect
the terminal bonus.

Training: best model found at step 160k (curriculum level 0) then policy degraded for the
remaining 840k steps. mean_reward = -217 with the final model.

### Root Cause

**Jerk and action_smooth were inside the velocity-scaled block:**

```python
# Before: ALL these are multiplied by v/v_max
core = w_heave*r_heave + w_wheel*r_wheel + w_accel*r_accel + w_jerk*r_jerk + w_smooth*r_smooth
total = (v/v_max) * core + tracking_penalty
```

At v = 13.6 km/h, velocity scale = 13.6/72 = 0.189.
The jerk penalty was discounted to 18.9% of its full value.

Per-step weighted contributions at 13.6 km/h:
```
r_tracking  : -0.208/step  (unscaled — full strength)
r_jerk      : -0.072/step  (scaled × 0.189 — cheap!)
r_smooth    : -0.006/step  (scaled × 0.189 — essentially free!)
```

The agent discovered: at low speed, generate enormous jerk freely, hover near rms=1.0,
occasionally collect terminal +100.

Per episode: jerk contributed only -20 (scaled), vs -108 it would cost at full speed.
The agent was paying 5× less for the same oscillation by going slowly.

**Why did the policy peak at 160k?**
Best model found in curriculum level 0 (0–350k steps). The easy level-0 roads allowed the
agent to find the oscillation strategy. After curriculum transitions, the harder roads caused
reward collapse because the agent had learned a strategy that only worked on easy roads.

### What We Tried

Reducing the default learning rate was considered. But the config already had Optuna-tuned
hyperparameters (`lr=8.5e-5`, `[512,512]` nets) which are already conservative. The
real fix was structural.

### Fix

Move `r_jerk` and `r_action_smooth` OUTSIDE the velocity-scaled block:

```python
# After: jerk/smooth unscaled — same cost at any speed
jerk_smooth = w_jerk*r_jerk + w_smooth*r_smooth          # unscaled
core = w_heave*r_heave + w_wheel*r_wheel + w_accel*r_accel  # scaled
total = (v/v_max) * core + tracking_penalty + jerk_smooth
```

Rationale: jerk and action_smooth measure self-induced longitudinal oscillation.
The agent fully controls these regardless of speed — oscillating at 13.6 km/h is just as
uncomfortable for passengers as oscillating at 72 km/h. There is no physical reason to
discount them at low speed (unlike heave, which physically depends on road excitation × speed).

### Verification

Simulated episode returns after fix:

```
exp_10's strategy (13.6 km/h, oscillating, rms=0.82):  -182   ← exploit destroyed
good: 36 km/h, smooth, rms=0.3:                        +30    ← clearly best
slow+smooth: 13.6 km/h, smooth, rms=0.3:               +23    ← still viable if careful
```

The oscillating-slow strategy loses 200 points vs the good crossing strategy.

### Lesson

**Velocity scaling applies to road-excitation discomfort, not to self-induced oscillation.**

Heave and wheel acceleration physically depend on road excitation × vehicle speed — slower
speed genuinely reduces them. Jerk and action_smooth are determined entirely by the agent's
command pattern and should cost the same at any speed. Scaling them with velocity gives the
agent a free pass to oscillate whenever it drives slowly.

Rule of thumb: any reward term that the agent can trivially set to zero by its own choice
(stop accelerating smoothly, stop jerking) should NOT be velocity-scaled.
Compare to r_tracking: also unscaled for the same reason — stopping to avoid it defeats
the purpose.

---

## Issue 6 — Weak Forward Movement Incentive (No Positive Progress Signal)

**Date:** 2026-06-08
**Context:** Post-mortem of exp_10 analysis; added alongside Issue 5 fix.

### Observation

After fixing Issue 5 (jerk/smooth unscaled), the reward landscape showed:

```
Stopped (v=0):                -280
Creep 9 km/h, rms=0:           +31
Good crossing 36 km/h, rms=0.3: +34
```

The gap between the creep exploit and a genuinely good policy was only **3 points**.
Every reward term was non-positive — the only incentive to move faster came from
avoiding the tracking penalty. With enough other penalties active, the agent could
settle at low speed where the gradient toward faster movement was too weak for PPO
to reliably escape.

### Root Cause

No positive signal existed for forward progress. All terms were penalties:
- r_tracking penalises being below v_max (negative)
- r_heave, r_accel, r_jerk penalise discomfort (negative)
- Terminal bonus fires at end (sparse, once per episode)

The agent could find marginal local optima where the comfort penalties exactly
balanced the tracking penalty at some low speed, with no positive gradient pulling
it toward higher speeds.

### Fix

Added `r_progress = v / v_max ∈ [0, 1]` — an unscaled, always-positive reward
for forward movement. Applied outside the velocity-scaled block, same as tracking.

```python
r_progress = v / v_max
total = scale * core + tracking_penalty + jerk_smooth_penalty + w_progress * r_progress
```

`w_progress = 0.2`. At full speed this contributes +0.2/step = +60 per episode.

### Verification

```
Stopped:                         -280   (unchanged)
Creep 9 km/h, rms=0:              +39   (+8 from progress)
Good crossing 36 km/h, rms=0.3:   +64   (+30 from progress)
Flat road 72 km/h, rms=0:        +160   (new theoretical max, was +100)
```

Gap between creep and good crossing: **3 → 25 points**. The gradient now
clearly favours faster, smoother crossings over low-speed hovering.

### Lesson

**Every reward function needs at least one always-positive term proportional to
the desired behaviour.** Pure penalty-based rewards can have flat regions where
many suboptimal strategies produce similar returns, giving PPO insufficient gradient
to escape local optima. A small positive reward directly proportional to the goal
(move forward) provides a consistent pull in the right direction and widens the gap
between good and bad strategies.

---

## Issue 7 — Road Position Drift: ODE and Observation See the Wrong Bump Location

**Date:** 2026-06-08
**Exp:** exp_12, speed_bump, 1M steps, curriculum, seed=69 — flat eval curve despite curriculum progression

### Symptom

Eval return curve oscillates without clear upward trend (−268 → −224 → −297 → −177 → −297 over 1M steps).
The preview wrapper correctly predicted upcoming bumps in the observation, but the agent was unable to
learn a consistent relationship between the preview signal and the bump disturbance it actually experienced.

### Root Cause

`RoadGenerator.get_height(t)` and `get_height_dot(t)` computed the road position as:

```python
x = self.speed * t   # current speed × elapsed time
```

The actual vehicle arc-length position is `s_pos = Σ vᵢ × DT`. Once the agent changes speed, these
two diverge permanently. Concrete example:

```
Agent starts at 20 m/s, brakes to 5 m/s over 3 seconds.
  actual s_pos  ≈ 52 m   (integrated over the deceleration)
  road.get_height(t=3) = 5 × 3 = 15 m   ← 37 m behind actual position
```

The `PreviewWrapper` used `env._s_pos` (the correct integrated position) to show upcoming bumps.
But `_obs()` called `road.get_height(self._t)` and `road.get_height_dot(self._t)` — wrong position.
The ODE sub-steps were also computed at `z_q_fn(t0 + i*DT_SIM)` which resolved to
`speed_current × (t0 + i*DT_SIM)` — again wrong.

**Net effect:** the preview showed a bump at the correct location, but the ODE forces arrived at a
completely different time. The correlation between observation and disturbance was broken, making
the preview signal effectively useless for learning.

### Fix

Added position-based query methods to `RoadGenerator`:

```python
def get_height_at(self, s: float) -> float:
    # bumps parameterised by arc-length s, not speed×time
    for x0, A, L in self._bumps:
        dx = s - x0
        if 0.0 <= dx <= L:
            return (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L))
    return 0.0

def get_height_dot_at(self, s: float, v: float) -> float:
    # ζ̇ = dζ/dx · v;  v passed explicitly, no implicit self.speed
    for x0, A, L in self._bumps:
        dx = s - x0
        if 0.0 < dx < L:
            dzdx = (A / 2.0) * (2.0 * np.pi / L) * np.sin(2.0 * np.pi * dx / L)
            return dzdx * v
    return 0.0
```

`QuarterCarODE.step()` signature changed from `(x, z_q_fn, t0)` to `(x, road, s_pos, v)`.
Sub-steps now sample the road at the correct arc-length position:

```python
for i in range(N_SUB):
    s0 = s_pos + i       * dt * v
    sh = s_pos + (i+0.5) * dt * v
    se = s_pos + (i+1.0) * dt * v
    zq_pre[i, 0] = road.get_height_dot_at(s0, v)
    ...
```

`_obs()` updated to use `road.get_height_at(self._s_pos)` and `road.get_height_dot_at(self._s_pos, self._v)`.

In `step()`, `s_pos_start` is captured before `self._s_pos` is incremented, so the ODE
sees the position at the start of the control step (not the end):

```python
s_pos_start   = self._s_pos
self._s_pos  += v_new * DT
new_state, z_B_ddot, z_W_ddot = self._ode.step(
    self._state, self._road, s_pos_start, v_road
)
```

### Lesson

**Never infer position from `speed × time` when speed is variable.**
The road is a spatial function `ζ(x)`. The correct query is always `ζ(s_pos)` where `s_pos` is the
integrated arc-length, not a time-domain approximation. The two are identical only at exactly
constant speed — the moment the agent does anything useful (braking before a bump), they diverge.

**All road queries must use the same position reference.**
Here, preview used `s_pos` (correct) while ODE used `speed × t` (wrong). Any inconsistency between
what the observation predicts and what the ODE delivers destroys the learning signal for those features.

---

## Issue 8 — `_max_distance` Frozen to Initial Road, Never Updated After Reset

**Date:** 2026-06-08
**Exp:** exp_12 post-mortem

### Symptom

Some training episodes appeared to terminate far too early (s_pos << last bump position).
Other episodes ran the full 300 steps on roads where all bumps were cleared at step 200.
Inconsistent episode structure made the terminal bonus fire at different road completion states.

### Root Cause

`_max_distance` was computed once in `__init__` from the static `MULTI_BUMP_CONFIG` road
(4 bumps, last ending at ~44 m → `_max_distance = 49 m`):

```python
# __init__: only called once
self._max_distance = self._compute_max_distance()   # = 49 m, frozen forever
```

`reset()` replaced `self._road` with a randomly generated road but never recomputed `_max_distance`.

**Two failure modes depending on random road length vs. 49 m:**

| Random road | `_max_distance` = 49 m | Effect |
|---|---|---|
| Level-0: 1–2 bumps, last at ~27 m | 49 m > 27 m → `road_complete` never fires | Full 300 steps always, even after road is done |
| Level-3: 5–6 bumps, last at ~60 m | 49 m < 60 m → `road_complete` fires early | Episode ends before last 2 bumps are reached |

The agent was rewarded/penalised for partial road traversal at inconsistent points, making it
impossible to learn a stable strategy across road lengths.

### Fix

One line added at the bottom of `reset()`, after the road is regenerated:

```python
self._max_distance = self._compute_max_distance()
```

`_compute_max_distance()` reads `self._road._bumps` (the freshly randomised bumps) so
it always returns the correct distance for the current episode's road layout.

### Lesson

**Any quantity derived from the road must be recomputed after the road changes.**
`_max_distance` was derived from the initial road but the road changes every episode.
The fix is trivially small — the cost of missing it was major inconsistency in episode structure.

---

## Issue 9 — Eval Always Ran at Full Difficulty While Training Used Curriculum

**Date:** 2026-06-08
**Exp:** exp_12 post-mortem

### Symptom

The eval return curve showed no clear trend despite training clearly progressing through
curriculum levels. The "best model" at step 160k (−111) was actually trained on level-0 roads
(4–7 cm bumps, 25–54 km/h) but evaluated on full-random roads (5–25 cm bumps, full speed range).
The eval curve was measuring a skill gap, not policy quality.

### Root Cause

`make_eval_vec_env()` was called without curriculum:

```python
# environment.py — before fix
eval_venv = make_eval_vec_env(road=eval_road, ..., curriculum_cfg=None)
```

The eval env used `road_params.yaml random` section: bump heights up to 0.25 m, 1–5 bumps,
up to v_max. Curriculum level 3 maxes out at 0.15 m, 2–6 bumps. So eval was always harder
than the hardest training level, for the entire 1M step run.

Variance in eval scores (same policy, different random roads) masked any real improvement signal.
The "best model" at 160k was the best because it happened to be evaluated on easier random roads
in that eval window — not because it was actually a better policy.

### Fix

**Two-part fix:**

**1. Curriculum level synchronisation** — eval env now wraps with `CurriculumWrapper` using
`set_forced_level()`. `VecNormalizeSyncCallback._on_step()` reads the training env's current
level via VecEnv attribute delegation and pushes it to the eval env:

```python
# monitoring.py
levels = self._train.venv.get_attr("current_level")
level  = int(levels[0]) if levels else 0
self._eval.venv.env_method("set_forced_level", level)
```

**2. `CurriculumWrapper.set_forced_level()`** — new method pins the wrapper to a specific
difficulty level, bypassing its internal step counter:

```python
def set_forced_level(self, level: Optional[int]) -> None:
    self._forced_level = level

def _current_level(self) -> int:
    if self._forced_level is not None:
        return min(self._forced_level, len(self._thresholds))
    ...  # normal step-count logic
```

`train.py` now passes `curriculum_cfg` to `make_eval_vec_env()`. Eval starts at level 0 and
advances in lockstep with training via the sync callback.

### Lesson

**Eval difficulty must match training difficulty to produce meaningful learning curves.**
An eval env that is always harder than the training curriculum measures the gap between current
skill and maximum difficulty — not whether the agent is improving. The noise from road randomisation
at a fixed hard difficulty can easily swamp the improvement signal from 200k training steps.

Match eval to training or use fixed evaluation scenarios (fixed-seed roads) to isolate
policy quality from road-difficulty variance.

---

## Issue 10 — Conflicting Reward: v_ref = v_max Creates Anti-Braking Gradient

**Date:** 2026-06-08
**Exp:** exp_12 post-mortem

### Symptom

Even after fixing the road-position bug (Issue 7), training with a constant `v_ref = v_max`
creates a conflict: `r_tracking + r_progress` both push the agent to stay near v_max, while
`r_heave + r_wheel` punish the body acceleration caused by hitting bumps at high speed.

The agent must discover by itself that slowing before bumps is worth it — but the tracking gradient
pushes back at every step below v_max, and `r_progress = v / v_max` also rewards going faster.
This makes anticipatory braking hard to learn.

### Root Cause

```python
def _compute_v_ref(self, t: float) -> float:
    return self._rcfg.v_max   # always 72 km/h, regardless of upcoming road
```

The reference profile was constant. Any speed below 72 km/h produced a tracking penalty,
even when the correct behaviour was to slow to 30 km/h before a 15 cm bump.

### Fix

`_compute_v_ref()` now looks ahead using `get_spatial_preview()` and reduces `v_ref` in
proportion to the height and proximity of the nearest upcoming significant bump:

```python
heights   = self._road.get_spatial_preview(s_pos=self._s_pos, ..., n_points=20)
max_h     = max(heights)
d         = distance to nearest peak ≥ peak_height_min
h_ratio   = clip(max_h / h_clip, 0, 1)      # 0 = tiny, 1 = tall
proximity = max(0, 1 − d / preview_distance)  # 0 = far, 1 = right here
v_ref     = v_max × (1 − 0.5 × h_ratio × proximity)
```

A 0.15 m bump (h_ratio = 1.0) at 5 m ahead (proximity = 0.75, preview = 20 m):
`v_ref = 20 × (1 − 0.5 × 0.75) = 12.5 m/s = 45 km/h`

The agent is now explicitly given a lower speed target near tall bumps, eliminating the
conflicting gradient between tracking and comfort near obstacles.

At 20 m distance (bump just entering preview horizon): `proximity = 0` → `v_ref = v_max` (no change).
The reduction only activates as the bump gets close, leaving the normal v_max target on flat road.

### Lesson

**A constant v_ref = v_max conflicts with any task that requires slowing down.**
The purpose of the tracking term is to keep the agent moving, not to force maximum speed at
all times. A speed reference that adapts to the road — lower near obstacles, high on flat — removes
the conflicting gradient and explicitly encodes the desired human-like trapezoidal speed profile
described in Mandl (2021): approach at v_max, reduce before bump, resume after.

---

## Issue 11 — No Reward for Actually Crossing Bumps

**Date:** 2026-06-08
**Exp:** exp_12 post-mortem

### Observation

All existing rewards were continuous per-step signals. There was no positive reward that fired
specifically when the agent crossed a bump. The agent could achieve good per-step returns by
staying on flat road (high `r_progress`, zero `r_heave`) without ever touching a bump.

The stop-and-wait exploit (Issue 1) and creep exploit (Issue 3) were symptoms of the same
underlying gap: no direct reward for the core task — navigating over the obstacles.

### Fix

Added `r_bumps`: a one-shot positive reward fired each time `s_pos` clears a bump end.
The reward accumulates if multiple bumps are passed in one step (rare but possible at high speed).

```python
# quarter_car_env.py step()
while (self._bumps_passed < len(self._bump_ends)
       and self._s_pos >= self._bump_ends[self._bumps_passed]):
    self._bumps_passed += 1
    r_bumps += cfg.w_bump_cross        # default: +5 per bump

reward += r_bumps
breakdown["r_bumps"] = r_bumps
```

`_bump_ends` (sorted list of `x0 + L` for each bump) is recomputed in `reset()` after the
random road is regenerated.

Weight `w_bump_cross = 5.0` — roughly a 25-step equivalent of `r_progress` at full speed.
Large enough to incentivise crossing but not so large it overrides the comfort signal.

### Lesson

**The core task should be directly rewarded.** For a bump-crossing agent, the task is
"cross bumps." Every existing reward was either a speed incentive or a comfort penalty —
none of them directly fired for crossing an obstacle. Adding a direct crossing reward closes
the loop: the agent now gets an explicit signal that it has achieved the primary objective,
not just secondary consequences of achieving it.

---

## Issue 12 — Step-Count Curriculum Promotes the Agent Before It Has Learned

**Date:** 2026-06-08
**Context:** Design review after exp_12 analysis

### Symptom

The original curriculum advanced levels based purely on elapsed training steps:

```yaml
thresholds:
  - 350_000   # → level 1 at 350k steps no matter what
  - 500_000   # → level 2 at 500k steps no matter what
  - 700_000   # → level 3 at 700k steps no matter what
```

In exp_12 the best model appeared at step 160k (curriculum level 0) and then performance
degraded for the remaining 840k steps. The agent was pushed into level-1 difficulty at
step 350k whether its eval return was −270 or +60. An agent still stuck at −270 on level-0
roads was suddenly facing taller, faster, more numerous bumps — a harder task it had
no foundation to handle.

### Root Cause

Step-count thresholds are a proxy for mastery. They hold when:
1. Training progresses monotonically (agents always improve over time), AND
2. All agents improve at the same rate.

Neither holds for PPO on this problem. The policy can plateau, collapse, or oscillate.
Pushing it into harder terrain during a plateau compounds the problem — the new difficulty
produces a chaotic signal that interferes with whatever partial skill the agent had built.

### Fix

Replaced step-count thresholds with **performance-gated advancement**:

```yaml
advance_return_threshold:
  0: -120.0   # leave level 0 once window mean ≥ -120
  1: -100.0   # leave level 1 once window mean ≥ -100
  2:  -80.0   # leave level 2 once window mean ≥  -80

advance_window: 3  # consecutive eval runs that must all meet threshold
```

`PerformanceCurriculumCallback` subclasses `EvalCallback` and intercepts each eval result.
It maintains a per-level list of eval returns and checks the rolling window:

```python
window = self._level_returns[level][-self._window:]
if len(window) >= self._window and np.mean(window) >= threshold:
    self._advance()   # set_level() on training env via env_method
```

`set_level()` is one-way — the agent cannot regress to an easier level once it has
earned a harder one. This prevents oscillation if performance drops temporarily.

`CurriculumWrapper` separates two concerns:
- `set_level(n)`        — permanent one-way advance (training env, called by callback)
- `set_forced_level(n)` — mirror level (eval env, called by VecNormalizeSyncCallback)

### Verification

Simulated advancement sequence:
```
Evals at level 0:  -150, -130, -115  → window mean = -131.7  < -120  → stay
Evals at level 0:  -118, -115, -110  → window mean = -114.3  ≥ -120  → advance to 1
Evals at level 1:  -102,  -99,  -97  → window mean =  -99.3  ≥ -100  → advance to 2
```

Level report printed at end of training:
```
Curriculum Level Performance
  Level   Evals      Mean      Best  Threshold  Status
  ------  -----  --------  --------  ---------  --------
  0           5    -125.6    -115.0     -120.0  advanced at step 60,000
  1           4    -102.0     -97.0     -100.0  advanced at step 140,000
  2           3     -75.0     -70.0      -80.0  advanced at step 200,000
  3           3     -40.0     -30.0      final  active  (unlocked 200,000)
```

Per-level stats are also written to `summary.json` under `curriculum.per_level`.

### Lesson

**Curriculum is a scaffolding tool — remove the scaffold only when the foundation is solid.**
Time-based advancement assumes the agent improves monotonically, which it doesn't.
Performance-gated advancement is self-pacing: a fast-learning agent advances quickly,
a struggling agent stays at the current difficulty until it actually masters it.

The cost is that an agent that never solves level 0 never advances — but that is exactly
the right behaviour. A policy that hasn't learned level-0 roads has nothing to gain from
level-1 roads and everything to lose.

---

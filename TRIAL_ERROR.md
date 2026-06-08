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

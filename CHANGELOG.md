# Changelog

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

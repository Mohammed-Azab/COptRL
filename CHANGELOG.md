# Changelog

All entries are one line. For design rationale see `WHY_WE_DO_THAT.md`.

---

- b6dd10f — remove all `enable_*` flags from reward config; disable terms via weight=0 instead
- b4d8da1 — preview obs: replace dist/width with t2r/freq; reduce pt1_tau 0.2→0.05s; re-enable speed-band with separate v_limit; add noise_active flag
- 34a0828 — remove dead reward code: tracking-disabled block, all always-true enable guards, else branches
- a2c1f90 — ensure 60 m flat start in all scenarios, eval, and random roads so agent always has braking room
- 88fd0bb — MPC: drop v_ref entirely; speed plans freely between v_min and v_max; cost matches reward exactly
- 607e75f — MPC: replace 7-state OCP with 3-state speed planner (v, s_pos, u_prev); fix v_ref aliasing via analytic nearest-bump search
- e72a6a8 — remove v_ref deque, push_history entry, and render map entry (dead code after v_ref removal)
- 98abc68 — remove v_ref from all time-series plots and render speed panel
- c9c571d — lower zeta_dot_limit 1.5→0.7 so geometry clamp enforces braking before all catalog bumps
- f53b233 — fix human driver: hold speed mid-bump, never accelerate during crossing; remove v_ref from baseline plots
- 1851fa2 — drop speed_error from observation vector; clean verbose comments
- 08846a3 — remove tracking penalty from reward; add speed_error hint to observation
- 2e41433 — add v_ref oscillation fix and training impact analysis to CHANGELOG
- d17495a — fix v_ref oscillation: replace 20-point sampler with analytic bump search in `_compute_v_ref`
- 4a6e6a4 — add --log-data flag to eval/driver_eval/mpc scripts; saves run.mat, .npz, run_info.json
- ae1d35a — render: add v and v_ref speed arrows to schematic panel
- a5e136f — docs: add step_bonus rationale to WHY_WE_DO_THAT
- 330fc40 — lower curriculum thresholds; fix reward range display
- abfae90 — reward: add step_bonus=0.5 to shift per-step range positive/negative
- 04995fe — apply Optuna results: lower LR, clip=0.1, smaller network, harder curriculum
- 2fe0143 — tune: seed 0→1000 to avoid overlap with training seed 42
- c6ffa0c — fix tune recipe: remove positional study arg that was swallowing flags
- e19971a — fix tune: timesteps, curriculum off, eval ordering, evaluate_policy call
- f80d36d — update configuration files docstring
- ba2b38b — tune PPO hyperparams: linear LR decay, larger rollout, entropy bonus
- d3d8961 — update README and changelog
- 2a2ac0d — fix curriculum thresholds and per-level model saving
- f2047ab — fix MPC cost and horizon
- c51f780 — fix reward config and add bumps display

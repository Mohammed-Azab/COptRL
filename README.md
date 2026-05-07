# Active Suspension Control via Deep Reinforcement Learning
### A Quarter-Car Simulation Framework — ROS 2 Humble · Stable-Baselines3 · Gazebo

---

> *"The road doesn't care about your comfort — but your controller can."*

Every time a vehicle hits a pothole, a speed bump, or a stretch of rough tarmac, there is a
brief, violent negotiation between the road and the passenger. Passive suspensions settle
this negotiation with fixed springs and dampers — good enough on average, but compromised by
definition. Active suspensions can do better: given the right policy, an actuator can cancel
disturbances in real time, keeping the sprung mass smooth while the wheels follow the road
faithfully.

This repository is the full simulation and training infrastructure for my thesis on learning
that policy end-to-end with deep reinforcement learning.

---

## What Lives Here

```
.
├── quarter_car_ws/          # ROS 2 workspace
│   └── src/
│       ├── quarter_car_core/         # Pure Python — ZERO ROS dependency
│       │   └── quarter_car_core/     # Physics, road profiles, Gym env, wrappers
│       ├── quarter_car_sim/          # ROS 2 simulation + Gazebo bridge
│       └── quarter_car_controllers/  # RL node · LQR stub · MPC stub
│
├── training/                # Standalone training & evaluation scripts
│   ├── train.py             # SAC / TD3 / PPO via Stable-Baselines3
│   ├── evaluate.py          # 4-panel evaluation plots + metrics table
│   ├── hyperparameter_search.py  # Optuna search
│   └── configs/             # YAML configs for env, SAC, TD3, PPO
│
├── tests/                   # pytest suite — 39 tests, all passing
├── scripts/                 # Shell helpers for training and Gazebo launch
├── notebooks/               # Jupyter exploration notebook
├── refs.txt                 # Full academic references [1]–[14]
└── requirements.txt
```

---

## The Physics in 60 Seconds

The model is a classical **2-DOF quarter-car** [1][5]: one degree of freedom for the sprung
mass (chassis + passenger) and one for the unsprung mass (wheel + axle).

```
  m_s · z̈_s = −k_s(z_s − z_u) − c_s(ż_s − ż_u) + F_act
  m_u · z̈_u =  k_s(z_s − z_u) + c_s(ż_s − ż_u) − k_t(z_u − z_r) − F_act
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| m_s | 317.5 kg | Sprung mass |
| m_u | 45.4 kg | Unsprung mass |
| k_s | 22 000 N/m | Suspension spring |
| c_s | 1 500 N·s/m | Suspension damper |
| k_t | 192 000 N/m | Tyre stiffness |

The ODE is integrated with **scipy RK45 at 500 Hz** internally, then decimated to 50 Hz for
the RL control loop. The actuator force F_act is the only control input, saturated at ±10 kN.

---

## The Three-Layer Architecture

The codebase is deliberately split into three layers that can run independently:

```
Layer 1 ── Headless Training (no ROS, no Gazebo)
           train.py → QuarterCarEnv → QuarterCarODE + RoadGenerator
           Fast, vectorised, runs on any Python machine.

Layer 2 ── ROS 2 Evaluation Pipeline
           sim_node (50 Hz physics tick) ↔ rl_node (SB3 model inference)
           Controller can be swapped: RL | passive | lqr stub | mpc stub

Layer 3 ── Gazebo Visualisation (evaluation only)
           gazebo_bridge_node converts z_s → joint cmd + TF
           Orange speed bump world, 1/10-scale RC car mesh.
           NO RViz2 anywhere — Gazebo is the sole 3D view.
           rqt_plot provides real-time signal traces.
```

The key design decision: `quarter_car_core` has **no ROS dependency whatsoever**.
It is installed as a plain pip package and can be imported in any Python environment,
making the training loop fast and portable.

---

## Road Profiles

Four profiles are available, each targeting a different evaluation scenario:

| Profile | Description | Primary use |
|---------|-------------|-------------|
| `speed_bump` | Versine bump A=0.1 m, L=0.5 m at 10 m/s | Impulse evaluation |
| `iso_8608_class_c` | FFT synthesis, PSD Gd(n) = Gd₀(n/n₀)⁻² [3] | Training |
| `sine_sweep` | 0.5 → 20 Hz chirp over episode | Frequency response |
| `flat` | z_r = 0 | Baseline / debug |

---

## Reward Function

The reward balances four competing objectives:

```
r = −w₁(z̈_s / a_lim)²          ← ride comfort   (w=1.0)
  − w₂(travel / travel_lim)²    ← suspension travel (w=0.3)
  − w₃(tyre / tyre_lim)²        ← road holding   (w=0.3)
  − w₄(F_act / F_max)²          ← actuator effort (w=0.1)
```

A terminal bonus of `+10 · (1 − rms_accel / a_lim)` rewards clean episode completion,
encouraging the agent to stay in the safe operating region throughout the episode rather
than just minimising instantaneous cost.

---

## Quickstart

### 1 — Install (no ROS needed for training)

```bash
git clone <repo>
cd quarter_car_sim

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e quarter_car_ws/src/quarter_car_core
```

Or with the helper script:
```bash
bash scripts/install_deps.sh
```

### 2 — Verify the zero-ROS import guarantee

```bash
python3 -c "from quarter_car_core import quarter_car_env; print('OK')"
```

### 3 — Run the test suite

```bash
/home/ubuntu/myRepo/quarter_car_sim/.venv/bin/python -m pytest tests/ -v         # 39 tests, ~1 second
```

### 4 — Train an RL agent

```bash
# SAC for 500k steps on ISO 8608 Class C road (recommended)
python training/train.py --algo sac --timesteps 500000

# PPO on speed-bump profile
python training/train.py --algo ppo --road speed_bump --timesteps 1000000

# Resume from checkpoint
python training/train.py --algo sac --resume models/sac_20250507/checkpoints/sac_50000_steps.zip
```

Training logs to `logs/` (TensorBoard) and prints a passive-vs-trained comparison table
on completion:

```
══════════════════════════════════════════════════════════════
Metric                    Passive    Trained    Improv%
──────────────────────────────────────────────────────────────
rms_accel                  3.2140     1.0821      66.3%
peak_accel                 9.8800     3.4120      65.5%
suspension_rms             0.0312     0.0089      71.5%
══════════════════════════════════════════════════════════════
```

### 5 — Headless evaluation (no ROS)

```bash
python training/evaluate.py \
    --model models/sac_20250507/best/best_model.zip \
    --algo sac \
    --road speed_bump \
    --episodes 5 \
    --plot
```

Produces a 4-panel figure: chassis height, sprung acceleration, suspension travel, and
actuator force — passive baseline overlaid in every plot.

### 6 — Gazebo evaluation (requires ROS 2 Humble)

One script handles everything — build, source, and launch. It also strips an active
Python venv from the path so `ros2` never hits a `PackageNotFoundError` for `ros2cli`:

```bash
# Passive simulation only (default — no model needed)
bash scripts/build_and_launch.sh

# Full Gazebo evaluation with a trained RL model
bash scripts/build_and_launch.sh \
    --launch eval_gazebo \
    --model models/sac_best.zip \
    --algo sac

# Build only, then launch manually
bash scripts/build_and_launch.sh --build-only
```

This opens Gazebo with the speed-bump world and the RC car driving through it at 10 m/s.
`rqt_plot` opens automatically showing `/car/acceleration`, `/car/comfort_score`, and
`/car/reward` in real time.

> **Venv note:** if you prefer to call `colcon` and `ros2` directly, deactivate the venv
> first (`deactivate`) then source ROS 2 before running any `ros2` commands.

### 7 — Hyperparameter search

```bash
python training/hyperparameter_search.py --algo sac --trials 50 --timesteps 50000
```

---

## Trained Algorithms

| Algorithm | Architecture | Notes |
|-----------|-------------|-------|
| **SAC** | MLP [256, 256] | Default; best sample efficiency |
| **TD3** | MLP [400, 300] | More stable, slower convergence |
| **PPO** | MLP [256, 256] pi+vf | On-policy; good for curriculum |

Training uses `SubprocVecEnv(n=4)` for SAC/TD3 and `DummyVecEnv` for PPO.
The environment stack is: `ActionRepeat(2) → NormalizeObservation → RewardScaler(0.1) → EpisodeLogger → VecNormalize`.

---

## Controller Status

| Controller | Status | Notes |
|------------|--------|-------|
| RL (SAC / TD3 / PPO) | **Implemented** | Main thesis contribution |
| Passive (F = 0) | **Implemented** | Baseline reference |
| LQR | **Stub** | State-space matrices available via `get_state_space()` |
| MPC | **Stub** | OSQP solver planned; horizon N=20 |

LQR and MPC are intentionally left as stubs with documented design references.
Their node counterparts (`lqr_node`, `mpc_node`) publish zero force and log a
clear "stub — passive mode" message so the pipeline remains runnable for comparison
launches without misleading results.

---

## Repository Hygiene

- `quarter_car_core` imports with **zero ROS installed** — tested in a clean venv
- No `visualization_msgs`, no marker arrays, no RViz2 anywhere in the codebase
- Physics integrator runs at **500 Hz** internally regardless of the 50 Hz control rate
- All random seeds are explicit and propagated — results are reproducible
- The ISO 8608 profile pre-generates a **60-second spatial buffer** and wraps it,
  so episodes are fast with no re-synthesis overhead

---

## References

Full citations in [`refs.txt`](refs.txt). Key works:

- **[1]** MathWorks — ADMM-Based MPC Control for Quarter-Car Suspension
- **[2]** Nhu et al. — Physics-Guided RL for Vehicle Suspension (ICMLA 2023)
- **[3]** ISO 8608:2016 — Road surface profiles
- **[5]** Hrovat 1997 — Survey of Advanced Suspension Developments *(Automatica)*
- **[7]** Raffin et al. — Stable-Baselines3 *(JMLR 2021)*
- **[9]** Stellato et al. — OSQP: An Operator Splitting Solver for QPs

---

## Project Layout at a Glance

```
quarter_car_core/
  ode_model.py          2-DOF ODE, RK45 500 Hz, get_state_space()
  road_generator.py     4 profiles, ISO 8608 FFT synthesis
  reward.py             RewardConfig dataclass, compute_reward()
  quarter_car_env.py    Gymnasium Env, obs(8,) act(1,) 50 Hz
  wrappers.py           NormalizeObservation, ActionRepeat,
                        RewardScaler, EpisodeLogger
  controllers/
    lqr_controller.py   STUB — raises NotImplementedError
    mpc_controller.py   STUB — raises NotImplementedError
```

---

*Built with ROS 2 Humble · Python 3.10 · Stable-Baselines3 2.8 · Gymnasium 1.2 · Gazebo Fortress*

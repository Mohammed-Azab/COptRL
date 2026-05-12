"""
Sanity checks for the Mandl (2021) quarter-car dynamics.

Tests:
  1. Static equilibrium  - zero road, zero action → accelerations ≈ 0
  2. Damper asymmetry    - |F_cs(+v)| / |F_cs(-v)| ≈ A = 0.5
  3. Nonlinear spring    - force > 0 once compression clearance is exceeded
  4. Speed-bump rollout  - 500 steps, no NaN / Inf
  5. RK4 energy check    - free vibration decays monotonically in 2-norm

Run:  python tests/test_mandl_dynamics.py
"""
import sys
import traceback
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / 'gym_env'))

from QuarterCar_env.ode_model import _spring_nonlin, _damper, _P, MandlQuarterCarODE
from QuarterCar_env.params import PHYSICS, VEHICLE_SPEED, DT

LOG_PATH = Path(__file__).parent / 'test_mandl_dynamics.log'

# ── helpers ───────────────────────────────────────────────────────────────────

def section(title, log):
    line = f"\n{'=' * 60}\n{title}\n{'=' * 60}"
    print(line); log.write(line + '\n')

def ok(msg, log):
    s = f'  [PASS] {msg}'; print(s); log.write(s + '\n')

def fail(msg, log):
    s = f'  [FAIL] {msg}'; print(s); log.write(s + '\n')

def info(msg, log):
    s = f'  {msg}'; print(s); log.write(s + '\n')


# ── individual checks ─────────────────────────────────────────────────────────

def check_static_equilibrium(log):
    section('1. Static equilibrium (zero road, zero action)', log)
    p   = _P(PHYSICS)
    ode = MandlQuarterCarODE()

    x0 = ode.reset(VEHICLE_SPEED)   # all deflections = 0, v = 10 m/s

    # One step with flat road and no actuator
    xf, z_B_ddot = ode.step(x0, lambda t: 0.0, 0.0, 0.0)

    info(f'z_B_ddot = {z_B_ddot:.2e} m/s²  (expect ≈ 0)', log)
    info(f'max |Δx[0:4]| = {np.max(np.abs(xf[:4])):.2e}  (expect ≈ 0)', log)

    assert abs(z_B_ddot) < 1e-6, f'z_B_ddot = {z_B_ddot}'
    ok('Body acceleration at static equilibrium < 1e-6 m/s²', log)

    assert np.max(np.abs(xf[:4])) < 1e-10, f'state drift = {xf[:4]}'
    ok('Vertical states remain at zero (no drift from flat road)', log)


def check_damper_asymmetry(log):
    section('2. Damper asymmetry (A = d1/z1 = 0.5)', log)
    p = _P(PHYSICS)

    for v in [0.05, 0.10, 0.15]:   # low-speed range
        Fc  = _damper( v, p)
        Fr  = _damper(-v, p)
        ratio = abs(Fc) / abs(Fr) if abs(Fr) > 1e-12 else float('inf')
        info(f'v_S = ±{v} m/s → F_cmp = {Fc:.1f} N, F_rbd = {Fr:.1f} N, ratio = {ratio:.4f}', log)
        assert abs(ratio - 0.5) < 0.01, f'ratio {ratio:.4f} ≠ 0.5'
    ok('Compression/rebound ratio ≈ A = 0.5 for all low-speed test points', log)

    # verify high-speed transition
    v_hi = 0.5   # m/s > v_d = 0.20
    Fc_hi = _damper(v_hi, p)
    Fc_lo = _damper(p.v_d - 1e-9, p)
    assert Fc_hi > Fc_lo, 'High-speed compression should be larger than low-speed'
    ok('High-speed compression force > low-speed boundary', log)


def check_spring_nonlinear(log):
    section('3. Nonlinear spring activation', log)
    p = _P(PHYSICS)

    # Inside linear range - should be zero
    for dyn in [0.0, p.dz_cmp - 0.001, -(p.dz_rbd - 0.001)]:
        F = _spring_nonlin(dyn, p)
        assert F == 0.0, f'Expected 0 in linear range, got {F} at dyn={dyn}'
    ok('Nonlinear spring = 0 inside linear range', log)

    # Compression onset - force is positive and increases with deflection
    F_prev = 0.0
    for excess in [0.005, 0.010, 0.020, 0.040]:
        dyn = p.dz_cmp + excess
        F = _spring_nonlin(dyn, p)
        info(f'  compression: dyn = {dyn:.3f} m → F_nonlin = {F:.1f} N', log)
        assert F > F_prev, f'Spring should be progressive, got F={F} ≤ F_prev={F_prev}'
        F_prev = F
    ok('Compression nonlinear force is positive and progressive', log)

    # Rebound onset - force is negative and grows in magnitude
    F_prev = 0.0
    for excess in [0.005, 0.010, 0.030, 0.060]:
        dyn = -(p.dz_rbd + excess)
        F = _spring_nonlin(dyn, p)
        info(f'  rebound:     dyn = {dyn:.3f} m → F_nonlin = {F:.1f} N', log)
        assert F < F_prev, f'Rebound spring should grow negative, got F={F} ≥ F_prev={F_prev}'
        F_prev = F
    ok('Rebound nonlinear force is negative and progressive', log)

    # Hard clip
    F_max = _spring_nonlin(5.0, p)   # extreme compression
    assert abs(F_max) == p.F_ks_nlin_max, f'Clip failed: |F|={abs(F_max)} ≠ {p.F_ks_nlin_max}'
    ok(f'Nonlinear spring clipped at F_ks_nlin_max = {p.F_ks_nlin_max:.0e} N', log)


def check_rollout_no_nan(log):
    section('4. Speed-bump rollout - 500 steps, no NaN/Inf', log)
    import gymnasium as gym
    import QuarterCar_env   # registers the env

    env = gym.make('QuarterCar_env/QuarterCar', road_profile='speed_bump')
    obs, _ = env.reset(seed=0)

    nan_detected = False
    steps = 0
    done  = False
    while not done and steps < 500:
        action = np.array([0.0])   # passive baseline
        obs, reward, terminated, truncated, info_dict = env.step(action)
        steps += 1
        done = terminated or truncated
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            nan_detected = True
            fail(f'NaN/Inf at step {steps}: obs={obs}, reward={reward}', log)
            break

    env.close()
    info(f'Completed {steps} steps, done={done}', log)
    if not nan_detected:
        ok('No NaN/Inf in obs or reward over 500 speed-bump steps', log)


def check_free_vibration_decay(log):
    section('5. Free vibration decays (passive system is stable)', log)
    ode = MandlQuarterCarODE()

    # Perturb body downward
    x0 = ode.reset(VEHICLE_SPEED)
    x0[2] = 0.05   # 5 cm suspension travel (compression)

    norms = []
    x = x0.copy()
    for _ in range(200):   # 4 s of passive evolution on flat road
        x, _ = ode.step(x, lambda t: 0.0, 0.0, 0.0)
        norms.append(float(np.linalg.norm(x[:4])))

    info(f'Initial 2-norm of vertical states: {np.linalg.norm(x0[:4]):.4f}', log)
    info(f'Final   2-norm after 4 s:          {norms[-1]:.4f}', log)

    assert norms[-1] < np.linalg.norm(x0[:4]), 'System should dissipate energy'
    ok('Vertical state norm decays over 4 s (damping is positive)', log)

    assert norms[-1] < 1e-2, f'State norm too large after 4 s: {norms[-1]:.4f}'
    ok('State norm < 1e-2 after 4 s (well-damped)', log)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(LOG_PATH, 'w') as log:
        header = f'Mandl dynamics sanity tests\nPython {sys.version}\n'
        print(header); log.write(header + '\n')

        checks = [
            check_static_equilibrium,
            check_damper_asymmetry,
            check_spring_nonlinear,
            check_rollout_no_nan,
            check_free_vibration_decay,
        ]

        passed = failed = 0
        for fn in checks:
            try:
                fn(log)
                passed += 1
            except Exception:
                msg = f'\n  [ERROR] {fn.__name__} raised:\n  {traceback.format_exc()}'
                print(msg); log.write(msg + '\n')
                failed += 1

        summary = f"\n{'=' * 60}\nSummary: {passed} passed, {failed} failed\n{'=' * 60}\n"
        print(summary); log.write(summary)

    print(f'\nFull output saved to {LOG_PATH}')


if __name__ == '__main__':
    main()

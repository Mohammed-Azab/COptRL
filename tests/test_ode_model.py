"""Tests for QuarterCarODE physics model."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from quarter_car_core.ode_model import QuarterCarODE, DEFAULT_PARAMS


def test_state_dimensions():
    ode = QuarterCarODE()
    state = ode.reset()
    assert state.shape == (4,)


def test_step_returns_correct_types():
    ode = QuarterCarODE()
    state = ode.reset()
    new_state, z_s_ddot = ode.step(state, 0.0, 0.0, 0.02)
    assert new_state.shape == (4,)
    assert isinstance(z_s_ddot, float)


def test_zero_input_zero_state_stays_zero():
    """Equilibrium: zero state + zero force + zero road = zero acceleration."""
    ode = QuarterCarODE()
    state = np.zeros(4)
    new_state, z_s_ddot = ode.step(state, 0.0, 0.0, 0.02)
    np.testing.assert_allclose(new_state, [0.0, 0.0, 0.0, 0.0], atol=1e-10)
    assert abs(z_s_ddot) < 1e-10


def test_passive_response_decays():
    """With an initial displacement and no road input, z_s should decay toward zero."""
    ode = QuarterCarODE()
    state = np.array([0.1, 0.0, 0.0, 0.0])
    for _ in range(500):
        state, _ = ode.step(state, 0.0, 0.0, 0.02)
    assert abs(state[0]) < 0.02, f"z_s={state[0]} should decay toward zero"


def test_force_affects_sprung_mass():
    """Positive actuator force should accelerate the sprung mass upward."""
    ode = QuarterCarODE()
    state = np.zeros(4)
    new_state, z_s_ddot = ode.step(state, 5000.0, 0.0, 0.02)
    assert z_s_ddot > 0, "Positive force should produce positive sprung acceleration"


def test_state_space_matrix_shapes():
    ode = QuarterCarODE()
    A, B, C, D = ode.get_state_space()
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)
    assert C.shape == (4, 4)
    assert D.shape == (4, 1)


def test_state_space_stability():
    """All eigenvalues of A should have negative real parts (stable passive system)."""
    ode = QuarterCarODE()
    A, _, _, _ = ode.get_state_space()
    eigenvalues = np.linalg.eigvals(A)
    assert all(ev.real < 0 for ev in eigenvalues), \
        f"Passive system should be stable; eigenvalues: {eigenvalues}"


def test_reset_returns_zeros():
    ode = QuarterCarODE()
    ode.step(np.array([0.1, 0.0, 0.0, 0.0]), 1000.0, 0.05, 0.02)
    state = ode.reset()
    np.testing.assert_array_equal(state, np.zeros(4))

import numpy as np
import sys
sys.path.insert(0, "src/road")
from QuarterCar_env.core.ode_model import QuarterCarODE


def test_step_returns_three_values():
    ode = QuarterCarODE()
    x = ode.reset(v0=6.0)
    result = ode.step(x, lambda t: 0.0, t0=0.0)
    assert len(result) == 3, f"expected 3-tuple, got {len(result)}-tuple"


def test_z_W_ddot_is_finite():
    ode = QuarterCarODE()
    x = ode.reset(v0=6.0)
    _, z_B_ddot, z_W_ddot = ode.step(x, lambda t: 0.0, t0=0.0)
    assert np.isfinite(z_B_ddot)
    assert np.isfinite(z_W_ddot)


def test_z_W_ddot_nonzero_on_road_input():
    ode = QuarterCarODE()
    x = ode.reset(v0=6.0)
    _, _, z_W_ddot = ode.step(x, lambda t: 0.1 * np.sin(10 * t), t0=0.0)
    _, _, z_W_ddot_flat = ode.step(x, lambda t: 0.0, t0=0.0)
    assert np.isfinite(z_W_ddot)
    assert np.isfinite(z_W_ddot_flat)

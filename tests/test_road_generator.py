"""Tests for RoadGenerator."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

from quarter_car_core.road_generator import RoadGenerator


class TestSpeedBump:
    def test_zero_before_bump(self):
        rg = RoadGenerator('speed_bump', 10.0)
        assert rg.get_height(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_zero_after_bump(self):
        rg = RoadGenerator('speed_bump', 10.0)
        t_after = 0.5 / 10.0 + 0.1
        assert rg.get_height(t_after) == pytest.approx(0.0, abs=1e-9)

    def test_peak_at_center(self):
        rg = RoadGenerator('speed_bump', 10.0)
        t_center = (0.5 / 2.0) / 10.0
        h = rg.get_height(t_center)
        assert h == pytest.approx(0.1, rel=0.01), f"Peak should be ~0.1m, got {h}"

    def test_height_positive_over_bump(self):
        rg = RoadGenerator('speed_bump', 10.0)
        for t in np.linspace(0.001, 0.049, 20):
            assert rg.get_height(t) >= 0.0

    def test_bump_times_length(self):
        rg = RoadGenerator('speed_bump', 10.0)
        times = rg.get_bump_times()
        assert len(times) == 3

    def test_dot_zero_outside_bump(self):
        rg = RoadGenerator('speed_bump', 10.0)
        assert rg.get_height_dot(0.0) == pytest.approx(0.0, abs=1e-9)
        assert rg.get_height_dot(1.0) == pytest.approx(0.0, abs=1e-9)


class TestISO8608:
    def test_buffer_populated(self):
        rg = RoadGenerator('iso_8608_class_c', 10.0)
        assert rg._iso_h is not None
        assert len(rg._iso_h) > 0

    def test_nonzero_variance(self):
        rg = RoadGenerator('iso_8608_class_c', 10.0)
        assert np.std(rg._iso_h) > 1e-4, "ISO profile should have non-trivial variance"

    def test_reset_reproducible_with_seed(self):
        rg = RoadGenerator('iso_8608_class_c', 10.0)
        rg.reset(seed=42)
        h1 = rg.get_height(1.0)
        rg.reset(seed=42)
        h2 = rg.get_height(1.0)
        assert h1 == pytest.approx(h2)

    def test_get_bump_times_empty(self):
        rg = RoadGenerator('iso_8608_class_c', 10.0)
        assert rg.get_bump_times() == []


class TestFlat:
    def test_always_zero(self):
        rg = RoadGenerator('flat', 10.0)
        for t in [0.0, 1.0, 5.0, 10.0]:
            assert rg.get_height(t) == 0.0
            assert rg.get_height_dot(t) == 0.0


class TestSineSweep:
    def test_within_amplitude(self):
        rg = RoadGenerator('sine_sweep', 10.0, {'sweep_amplitude': 0.02})
        for t in np.linspace(0, 10, 100):
            assert abs(rg.get_height(t)) <= 0.02 + 1e-9

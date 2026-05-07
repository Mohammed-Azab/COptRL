"""Pytest configuration and shared fixtures for quarter-car RL tests."""
import sys
from pathlib import Path

# Ensure the quarter_car_core package is importable for all test modules
sys.path.insert(0, str(Path(__file__).parents[1] / 'quarter_car_ws' / 'src' / 'quarter_car_core'))

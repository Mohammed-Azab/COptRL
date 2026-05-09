# Road profile generator for quarter-car simulation.
# Profiles: speed_bump, iso_8608_class_c, sine_sweep, flat.
import numpy as np

from quarter_car_core.params import ROAD_DEFAULTS


class RoadGenerator:
    def __init__(self, profile: str = 'speed_bump', vehicle_speed: float = 10.0,
                 params: dict = None):
        self.profile = profile
        self.speed = vehicle_speed
        p = {**ROAD_DEFAULTS, **(params or {})}

        self._bump_A = p['bump_height']     # m
        self._bump_L = p['bump_length']     # m

        # ISO 8608 buffer is pre-generated at init and regenerated on reset()
        self._iso_Gd0      = p['iso_gd0']
        self._iso_n0       = p['iso_n0']
        self._iso_dt       = 0.002   # s — internal integration step
        self._iso_duration = 60.0   # s — pre-generated buffer length

        self._sweep_A     = p['sweep_amplitude']
        self._sweep_f_min = 0.5    # Hz
        self._sweep_f_max = 20.0   # Hz
        self._ep_duration = p['episode_duration']

        self._iso_h     = None
        self._iso_h_dot = None
        self._build_iso_buffer(seed=None)

    # ------------------------------------------------------------------
    def _build_iso_buffer(self, seed=None):
        rng = np.random.default_rng(seed)
        n    = int(self._iso_duration / self._iso_dt)
        dx   = self.speed * self._iso_dt
        x_total = n * dx

        n_freq      = np.fft.rfftfreq(n, d=dx)
        n_freq[0]   = 1e-9  # avoid divide-by-zero at DC
        Gd          = self._iso_Gd0 * (n_freq / self._iso_n0) ** (-2)
        # IRFFT divides by n, so multiply by n/2 to recover the correct physical amplitude.
        # The factor sqrt(2) gives one-sided → two-sided PSD conversion.
        amp         = (n / 2.0) * np.sqrt(2.0 * Gd / x_total)
        phases      = rng.uniform(0, 2 * np.pi, size=len(n_freq))
        spectrum    = amp * np.exp(1j * phases)
        spectrum[0] = 0.0   # zero mean

        h = np.fft.irfft(spectrum, n=n)
        self._iso_h     = h
        self._iso_h_dot = np.gradient(h, self._iso_dt)

    # ------------------------------------------------------------------
    def get_height(self, t: float) -> float:
        if self.profile == 'flat':
            return 0.0
        if self.profile == 'speed_bump':
            x = self.speed * t
            if 0.0 <= x <= self._bump_L:
                return (self._bump_A / 2.0) * (1.0 - np.cos(2.0 * np.pi * x / self._bump_L))
            return 0.0
        if self.profile == 'iso_8608_class_c':
            idx = int(t / self._iso_dt) % len(self._iso_h)
            return float(self._iso_h[idx])
        if self.profile == 'sine_sweep':
            ratio = min(t / self._ep_duration, 1.0)
            f = self._sweep_f_min + (self._sweep_f_max - self._sweep_f_min) * ratio
            return self._sweep_A * np.sin(2.0 * np.pi * f * t)
        return 0.0

    def get_height_dot(self, t: float) -> float:
        if self.profile == 'flat':
            return 0.0
        if self.profile == 'speed_bump':
            x = self.speed * t
            if 0.0 < x < self._bump_L:
                dzdx = (self._bump_A / 2.0) * (2.0 * np.pi / self._bump_L) * np.sin(
                    2.0 * np.pi * x / self._bump_L)
                return dzdx * self.speed
            return 0.0
        if self.profile == 'iso_8608_class_c':
            idx = int(t / self._iso_dt) % len(self._iso_h_dot)
            return float(self._iso_h_dot[idx])
        if self.profile == 'sine_sweep':
            eps = 1e-5
            return (self.get_height(t + eps) - self.get_height(t - eps)) / (2.0 * eps)
        return 0.0

    def reset(self, seed=None):
        if self.profile == 'iso_8608_class_c':
            self._build_iso_buffer(seed=seed)

    def get_bump_times(self) -> list:
        if self.profile != 'speed_bump':
            return []
        return [
            0.0,
            (self._bump_L / 2.0) / self.speed,
            self._bump_L / self.speed,
        ]

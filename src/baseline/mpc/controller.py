import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from QuarterCar_env.config.env_params import PHYSICS, DT
from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config
from road.road_generator import RoadGenerator
from ocp import build_solver


# v_ref preview — mirrors env._compute_v_ref but position-based

def _v_ref_seq(s_pos: float, v0: float, bumps: list, cfg: RewardConfig, N: int, dt: float) -> np.ndarray:
    out = np.empty(N)
    s   = s_pos
    v   = v0
    for k in range(N):
        out[k] = _v_ref_at(s, bumps, cfg)
        v  = float(np.clip(v, 0.0, cfg.v_max))
        s += v * dt
    return out


def _v_ref_at(s_pos: float, bumps: list, cfg: RewardConfig) -> float:
    if not bumps:
        return cfg.v_max
    preview = cfg.preview_distance
    n = 20
    s_pts   = s_pos + np.linspace(preview / n, preview, n)
    heights = np.zeros(n)
    for x0, A, L in bumps:
        dx = s_pts - x0
        mask = (dx >= 0.0) & (dx <= L)
        heights[mask] += (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx[mask] / L))
    max_h = float(np.max(heights))
    if max_h < cfg.peak_height_min:
        return cfg.v_max
    for i, h in enumerate(heights):
        if h >= cfg.peak_height_min:
            d = (i + 1) * preview / n
            break
    else:
        return cfg.v_max
    h_ratio   = min(1.0, max_h / cfg.h_clip)
    proximity = max(0.0, 1.0 - d / preview)
    return max(cfg.v_min, cfg.v_max * (1.0 - 0.5 * h_ratio * proximity))


# bump geometry hash — used to detect road changes

_OCP_VERSION = 'v3'   # bump when cost structure changes (nr/nr_e dimensions)

def _bumps_hash(bumps: list) -> str:
    key = (_OCP_VERSION + str(sorted(bumps))).encode()
    return hashlib.md5(key).hexdigest()[:8]


@contextmanager
def _suppress_native_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


# MPC controller

class MPCController:
    def __init__(
        self,
        cfg:                  Optional[RewardConfig] = None,
        physics:              Optional[dict] = None,
        N:                    int   = 50,
        dt:                   float = DT,
        nlp_solver_max_iter:  int   = 10,
    ):
        self._cfg     = cfg or load_reward_config()
        self._physics = dict(physics or PHYSICS)
        # inject cfg limits so ocp.py can reference them via physics dict
        self._physics['v_max'] = float(self._cfg.v_max)
        self._physics['a_max'] = float(self._cfg.a_max)

        self._N                   = N
        self._dt                  = dt
        self._nlp_solver_max_iter = nlp_solver_max_iter
        self._solver              = None
        self._bumps_hash          = None
        self._bumps: list         = []
        self._prev_u              = 0.0
        self._gen_base            = Path(tempfile.gettempdir()) / 'acados_qc'

    def reset(self, road: RoadGenerator) -> None:
        bumps = list(road._bumps) if road.profile == 'speed_bump' else []
        bh    = _bumps_hash(bumps)
        if bh != self._bumps_hash:
            # rebuild (and recompile) solver for this road geometry
            gen_dir = str(self._gen_base / bh)
            os.makedirs(gen_dir, exist_ok=True)
            with _suppress_native_output():
                self._solver = build_solver(
                    self._physics, bumps, self._cfg,
                    N=self._N, dt=self._dt, gen_dir=gen_dir,
                    nlp_solver_max_iter=self._nlp_solver_max_iter,
                )
            self._bumps_hash = bh
            self._bumps      = bumps
        self._prev_u = 0.0

    def act(self, x: np.ndarray, s_pos: float, road: RoadGenerator) -> float:
        # hot-path: road must have been set via reset() before calling act()
        if self._solver is None:
            self.reset(road)

        N      = self._N
        dt     = self._dt
        solver = self._solver

        # augmented state: append s_pos as x[6]
        x_aug = np.append(x.astype(float), float(s_pos))

        # set initial state
        solver.set(0, 'lbx', x_aug)
        solver.set(0, 'ubx', x_aug)

        # precompute v_ref along nominal trajectory and set as online parameter
        v_ref_seq = _v_ref_seq(s_pos, float(x[4]), self._bumps, self._cfg, N, dt)
        for k in range(N):
            solver.set(k, 'p', np.array([v_ref_seq[k]]))
        solver.set(N, 'p', np.array([v_ref_seq[-1]]))

        with _suppress_native_output():
            status = solver.solve()
        # 0 = success, 2 = max_iter (partial solve, still usable)
        if status not in (0, 2):
            return self._prev_u   # keep last action on solver failure

        u0 = float(solver.get(0, 'u')[0])
        u0 = float(np.clip(u0, -1.0, 1.0))
        self._prev_u = u0
        return u0

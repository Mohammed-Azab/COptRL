import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from QuarterCar_env.config.env_params import PHYSICS, DT
from QuarterCar_env.config.reward_params import RewardConfig, load_reward_config
from road.road_generator import RoadGenerator
from ocp import build_solver_simple


_OCP_VERSION = 'v5'   # bump when cost structure changes (nr/nr_e dimensions)


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
        self._physics['v_max'] = float(self._cfg.v_max)
        self._physics['a_max'] = float(self._cfg.a_max)

        self._N                   = N
        self._dt                  = dt
        self._nlp_solver_max_iter = nlp_solver_max_iter
        self._prev_u              = 0.0
        self._gen_base            = Path(tempfile.gettempdir()) / 'acados_qc'

        # 3-state speed-planner: [v, s_pos, u_prev]. No suspension dynamics —
        # the full 7-state model causes solver failures from bumpstop Jacobians.
        gen_dir = str(self._gen_base / (_OCP_VERSION + '_simple'))
        os.makedirs(gen_dir, exist_ok=True)
        with _suppress_native_output():
            self._solver = build_solver_simple(
                self._physics, self._cfg,
                N=self._N, dt=self._dt, gen_dir=gen_dir,
                nlp_solver_max_iter=self._nlp_solver_max_iter,
            )

    def reset(self, road: RoadGenerator) -> None:
        self._prev_u = 0.0

    def act(self, x: np.ndarray, s_pos: float, road: RoadGenerator) -> float:
        if self._solver is None:
            self.reset(road)

        solver = self._solver

        # 3-state initial condition: [v, s_pos, u_prev]
        x_init = np.array([float(x[4]), float(s_pos), self._prev_u])
        solver.set(0, 'lbx', x_init)
        solver.set(0, 'ubx', x_init)

        with _suppress_native_output():
            status = solver.solve()
        if status not in (0, 2):
            return self._prev_u

        u0 = float(np.clip(float(solver.get(0, 'u')[0]), -1.0, 1.0))
        self._prev_u = u0
        return u0

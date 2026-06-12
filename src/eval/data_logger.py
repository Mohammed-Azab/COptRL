"""
Run-data logger for eval.py, driver_eval.py, and mpc.py.

Saves per-episode timeseries to  data/<method>/<road>/<run_id>/
  run.mat   MATLAB struct (all arrays as max_steps × n_episodes matrices)
  run.npz   NumPy archive  (same arrays, same layout)
  run_info.json human-readable metadata + variable guide

Layout (both formats)
---------------------
Timeseries: shape (max_steps, n_episodes), NaN-padded for shorter episodes:
  t           time axis                 [s]
  v_kmh       longitudinal speed        [km/h]
  v_ref_kmh   speed reference           [km/h]
  z_B_ddot    body vertical accel       [m/s²]
  z_W_ddot    wheel vertical accel      [m/s²]   (when available)
  action      normalised control input  [-1, 1]
  reward      per-step reward           [-]
  s_pos       arc-length position       [m]       (when available)

Per-episode scalars, shape (n_episodes,):
  episode_return    total episode return
  rms_accel         RMS body accel       [m/s²]
  comfort_score     comfort metric       [0-1]
  n_steps           valid steps before NaN padding

Metadata scalars (prefix meta_):
  meta_method       e.g. 'PPO', 'MPC', 'HumanDriver'
  meta_road         e.g. 'speed_bump'
  meta_n_episodes   int
  meta_dt           simulation timestep  [s]
  meta_v_max_kmh    speed limit          [km/h]
  meta_a_comfort    body-accel comfort threshold  [m/s²]
  meta_a_limit      terminal-bonus threshold      [m/s²]
  meta_run_id       experiment tag, e.g. exp_27 or run_1

MATLAB quick-start
------------------
    d = load('run.mat');
    % overlay all episodes
    figure; plot(d.t, d.v_kmh); xlabel('t [s]'); ylabel('v [km/h]');
    % per-episode mean speed
    mean_v = nanmean(d.v_kmh);
    % single episode
    ep = 1;
    valid = ~isnan(d.t(:, ep));
    plot(d.t(valid, ep), d.z_B_ddot(valid, ep));

Python quick-start
------------------
    import numpy as np
    d = np.load('run.npz')          # no allow_pickle needed, no object arrays
    t, v = d['t'], d['v_kmh']      # (max_steps, n_ep)
    # episode 0
    mask = ~np.isnan(t[:, 0])
    plt.plot(t[mask, 0], v[mask, 0])
    # string metadata lives in run_info.json, not run.npz
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_arr(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) if v is not None else np.empty(0)


def _pad_episodes(cols: list[np.ndarray]) -> np.ndarray:
    """Stack variable-length 1-D arrays → 2-D (max_len, n_ep), NaN-padded."""
    if not cols:
        return np.empty((0, 0))
    max_len = max(len(c) for c in cols)
    out = np.full((max_len, len(cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(cols):
        out[:len(c), j] = c
    return out


# ── public API ────────────────────────────────────────────────────────────────

class EpisodeData:
    """Accumulate one episode's timeseries then hand it to RunLogger."""

    __slots__ = ('v', 'v_ref', 'z_B_ddot', 'z_W_ddot',
                 'action', 'reward', 's_pos',
                 'episode_return', 'rms_accel', 'comfort_score')

    def __init__(
        self,
        v:              list | np.ndarray,
        v_ref:          list | np.ndarray,
        z_B_ddot:       list | np.ndarray,
        action:         list | np.ndarray,
        episode_return: float,
        rms_accel:      float,
        comfort_score:  float,
        z_W_ddot:       Optional[list | np.ndarray] = None,
        reward:         Optional[list | np.ndarray] = None,
        s_pos:          Optional[list | np.ndarray] = None,
    ):
        self.v              = _to_arr(v)
        self.v_ref          = _to_arr(v_ref)
        self.z_B_ddot       = _to_arr(z_B_ddot)
        self.z_W_ddot       = _to_arr(z_W_ddot) if z_W_ddot is not None else None
        self.action         = _to_arr(action)
        self.reward         = _to_arr(reward)   if reward    is not None else None
        self.s_pos          = _to_arr(s_pos)    if s_pos     is not None else None
        self.episode_return = float(episode_return)
        self.rms_accel      = float(rms_accel)
        self.comfort_score  = float(comfort_score)

    @property
    def n_steps(self) -> int:
        return len(self.v)


class RunLogger:
    """Collect episodes and save to data/."""

    def __init__(
        self,
        method:      str,
        road:        str,
        out_root:    Path,
        dt:          float = 0.02,
        v_max_kmh:   float = 72.0,
        a_comfort:   float = 3.0,
        a_limit:     float = 5.0,
        run_id:      str | None = None,
    ):
        self.method    = method
        self.road      = road
        self.dt        = dt
        self.v_max_kmh = v_max_kmh
        self.a_comfort = a_comfort
        self.a_limit   = a_limit
        self._episodes: list[EpisodeData] = []

        if run_id is None:
            parent = out_root / method / road
            existing: set[int] = set()
            if parent.is_dir():
                for d in parent.iterdir():
                    m = re.match(r'run_(\d+)$', d.name)
                    if m:
                        existing.add(int(m.group(1)))
            n = 1
            while n in existing:
                n += 1
            run_id = f'run_{n}'
        self.run_id   = run_id
        self.save_dir = out_root / method / road / self.run_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, ep: EpisodeData) -> None:
        self._episodes.append(ep)

    def save(self) -> dict[str, Path]:
        eps = self._episodes
        n   = len(eps)
        if n == 0:
            return {}

        dt = self.dt

        # ── build arrays ──────────────────────────────────────────────────
        arrays: dict[str, np.ndarray] = {}

        # time axis
        arrays['t'] = _pad_episodes([np.arange(e.n_steps, dtype=np.float64) * dt
                                     for e in eps])

        # timeseries channels
        arrays['v_kmh']     = _pad_episodes([e.v     * 3.6 for e in eps])
        arrays['v_ref_kmh'] = _pad_episodes([e.v_ref * 3.6 for e in eps])
        arrays['z_B_ddot']  = _pad_episodes([e.z_B_ddot    for e in eps])
        arrays['action']    = _pad_episodes([e.action       for e in eps])

        if any(e.z_W_ddot is not None and len(e.z_W_ddot) > 0 for e in eps):
            arrays['z_W_ddot'] = _pad_episodes([
                e.z_W_ddot if e.z_W_ddot is not None else np.array([]) for e in eps
            ])
        if any(e.reward is not None and len(e.reward) > 0 for e in eps):
            arrays['reward'] = _pad_episodes([
                e.reward if e.reward is not None else np.array([]) for e in eps
            ])
        if any(e.s_pos is not None and len(e.s_pos) > 0 for e in eps):
            arrays['s_pos'] = _pad_episodes([
                e.s_pos if e.s_pos is not None else np.array([]) for e in eps
            ])

        # per-episode scalars
        arrays['episode_return'] = np.array([e.episode_return for e in eps])
        arrays['rms_accel']      = np.array([e.rms_accel      for e in eps])
        arrays['comfort_score']  = np.array([e.comfort_score  for e in eps])
        arrays['n_steps']        = np.array([e.n_steps        for e in eps], dtype=np.float64)

        # numeric metadata only, no object arrays; npz loads without allow_pickle
        meta_num = {
            'meta_dt':         float(dt),
            'meta_v_max_kmh':  float(self.v_max_kmh),
            'meta_a_comfort':  float(self.a_comfort),
            'meta_a_limit':    float(self.a_limit),
            'meta_n_episodes': float(n),
        }
        # string metadata goes to JSON only (see companion below)
        meta_str = {
            'meta_method':  self.method,
            'meta_road':    self.road,
            'meta_run_id':  self.run_id,
        }

        saved: dict[str, Path] = {}

        # .npz: pure float arrays, no object dtype
        npz_path = self.save_dir / 'run.npz'
        np.savez(str(npz_path), **arrays, **meta_num)
        saved['npz'] = npz_path

        # ── .mat ──────────────────────────────────────────────────────────
        try:
            from scipy.io import savemat
            mat_dict = dict(arrays)
            mat_dict.update(meta_num)
            mat_dict.update(meta_str)   # savemat handles strings natively
            mat_path = self.save_dir / 'run.mat'
            savemat(str(mat_path), mat_dict, do_compression=True)
            saved['mat'] = mat_path
        except ImportError:
            pass   # scipy optional; .npz is always saved

        # ── companion JSON ─────────────────────────────────────────────────
        info = {
            'metadata': {**meta_str, **meta_num},
            'summary': {
                'mean_return':  float(np.mean([e.episode_return for e in eps])),
                'std_return':   float(np.std( [e.episode_return for e in eps])),
                'max_return':   float(np.max( [e.episode_return for e in eps])),
                'min_return':   float(np.min( [e.episode_return for e in eps])),
                'mean_rms_accel': float(np.mean([e.rms_accel for e in eps])),
                'mean_comfort':   float(np.mean([e.comfort_score for e in eps])),
            },
            'arrays': {
                k: {'shape': list(v.shape), 'unit': _UNITS.get(k, '?')}
                for k, v in arrays.items()
            },
            'variable_guide': _GUIDE,
        }
        json_path = self.save_dir / 'run_info.json'
        json_path.write_text(json.dumps(info, indent=2))
        saved['json'] = json_path

        return saved


# ── variable documentation ────────────────────────────────────────────────────

_UNITS: dict[str, str] = {
    't':              's',
    'v_kmh':          'km/h',
    'v_ref_kmh':      'km/h',
    'z_B_ddot':       'm/s²',
    'z_W_ddot':       'm/s²',
    'action':         '[-1, 1]',
    'reward':         'reward',
    's_pos':          'm',
    'episode_return': 'reward',
    'rms_accel':      'm/s²',
    'comfort_score':  '[0-1]',
    'n_steps':        'steps',
}

_GUIDE = {
    'timeseries_shape': '(max_steps, n_episodes) — NaN-padded for shorter episodes',
    'access_episode':   'd["t"][:, i]  or  d.t(:, i+1) in MATLAB  (0-indexed Python, 1-indexed MATLAB)',
    'valid_mask_python': 'mask = ~np.isnan(d["t"][:, ep_i])',
    'valid_mask_matlab': 'valid = ~isnan(d.t(:, ep));',
    'mean_over_eps':    'nanmean(d.v_kmh, 2) in MATLAB  /  np.nanmean(d["v_kmh"], axis=1) in Python',
}

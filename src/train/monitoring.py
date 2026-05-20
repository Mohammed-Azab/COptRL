from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize


class QuarterCarMetricsCallback(BaseCallback):
    """
    Reads ride-comfort and speed-tracking metrics from env info dicts and
    writes them to TensorBoard at every step.
    """

    def _on_step(self) -> bool:
        infos: list[dict] = self.locals.get("infos", [])
        comfort, rms_a, spd_err, ep_reward = [], [], [], []

        for info in infos:
            if "comfort_score" in info:
                comfort.append(info["comfort_score"])
            if "rms_accel" in info:
                rms_a.append(info["rms_accel"])
            if "speed_error" in info:
                spd_err.append(abs(info["speed_error"]))
            if "episode_reward" in info:
                ep_reward.append(info["episode_reward"])

        if comfort:
            self.logger.record("env/comfort_score",  float(np.mean(comfort)))
        if rms_a:
            self.logger.record("env/rms_accel_ms2",  float(np.mean(rms_a)))
        if spd_err:
            self.logger.record("env/speed_error_ms", float(np.mean(spd_err)))
        if ep_reward:
            self.logger.record("env/ep_reward", float(np.mean(ep_reward)))
        return True


class VecNormalizeSyncCallback(BaseCallback):

    def __init__(self, train_venv: VecNormalize, eval_venv: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self._train = train_venv
        self._eval  = eval_venv

    def _on_step(self) -> bool:
        self._eval.obs_rms = self._train.obs_rms
        self._eval.ret_rms = self._train.ret_rms
        return True


def build_callbacks(
    model_dir: Path,
    eval_venv: VecNormalize,
    train_venv: VecNormalize,
    eval_freq: int,
    n_eval_episodes: int,
    checkpoint_freq: int,
) -> CallbackList:
    #Assemble all callbacks into a single CallbackList
    best_model_path  = model_dir / "best"
    ckpt_path        = model_dir / "checkpoints"
    best_model_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=str(best_model_path),
        log_path=str(best_model_path),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(ckpt_path),
        name_prefix="ckpt",
        save_vecnormalize=True,
    )
    metrics_cb = QuarterCarMetricsCallback()
    sync_cb    = VecNormalizeSyncCallback(train_venv, eval_venv)

    return CallbackList([eval_cb, ckpt_cb, metrics_cb, sync_cb])

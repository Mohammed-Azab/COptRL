"""
Dual-branch features extractor for the QuarterCar preview observation.

Observation layout: [ state (6 scalars) | peak preview (n_peaks * 3) ]

  State branch   — MLP over the 6 scalar physics/speed features.
  Preview branch — MLP over the n_peaks * 3 peak-encoded features.
                   Each peak is [dist, height, width], so a flat MLP is the
                   right inductive bias here — no ordering assumption across peaks.

The two branches are concatenated and projected to features_dim.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# scalar features before the preview block: ζ, ζ̇, v/v_max, filtered_a, filtered_jerk, prev_action
_N_STATE_SCALARS = 6


class PreviewFeaturesExtractor(BaseFeaturesExtractor):
    """
    Split-stream extractor: MLP for state scalars, MLP for peak preview.

    Args:
        observation_space:   Gymnasium Box space for the full observation.
        n_preview_points:    Total preview feature count = n_peaks * 3.
        state_hidden_dim:    Width of the state MLP hidden layer.
        preview_hidden_dim:  Width of the preview MLP hidden layer.
        features_dim:        Output dimension fed to policy/value heads.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_preview_points: int = 9,
        state_hidden_dim: int = 64,
        preview_hidden_dim: int = 32,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        obs_dim = int(observation_space.shape[0])
        self._n_state   = obs_dim - n_preview_points
        self._n_preview = n_preview_points

        if self._n_state <= 0:
            raise ValueError(
                f"n_preview_points={n_preview_points} must be < obs_dim={obs_dim}"
            )

        self.state_branch = nn.Sequential(
            nn.Linear(self._n_state, state_hidden_dim),
            nn.ReLU(),
        )

        self.preview_branch = nn.Sequential(
            nn.Linear(self._n_preview, preview_hidden_dim),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(state_hidden_dim + preview_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state   = obs[:, : self._n_state]
        preview = obs[:, self._n_state :]

        return self.combine(
            torch.cat([self.state_branch(state), self.preview_branch(preview)], dim=1)
        )

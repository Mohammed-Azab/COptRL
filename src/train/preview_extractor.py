"""
Dual-branch features extractor for the QuarterCar preview observation.

The observation vector is split at the boundary between state features and
road-preview features:

    obs = [ state (obs_dim - n_preview_points) | preview (n_preview_points) ]

  • State branch  — small MLP; captures current physics state + speed context.
  • Preview branch — 1D conv over the n_preview_points spatial samples;
                     captures spatial patterns (bump approaching, bump width,
                     bump amplitude) that a flat MLP would only learn slowly
                     because it has no inductive bias for ordered sequences.

The two branches are concatenated and projected to `features_dim`, which is
then fed into SB3's standard policy/value heads (net_arch layers).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PreviewFeaturesExtractor(BaseFeaturesExtractor):
    """
    Split-stream extractor: MLP for state, 1D Conv for road preview.

    Args:
        observation_space:  Gymnasium Box space for the full observation.
        n_preview_points:   Number of preview height samples at the END of obs.
        state_hidden_dim:   Width of the state MLP hidden layer.
        conv_channels:      Number of Conv1d output channels.
        features_dim:       Dimension of the combined output fed to policy heads.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_preview_points: int = 10,
        state_hidden_dim: int = 64,
        conv_channels: int = 8,
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

        # State branch: single hidden layer MLP
        self.state_branch = nn.Sequential(
            nn.Linear(self._n_state, state_hidden_dim),
            nn.ReLU(),
        )

        # Preview branch: 1D conv → captures spatial bump profile
        # Input: (batch, 1, n_preview_points)
        # Output after flatten: (batch, conv_channels * n_preview_points)
        conv_flat_dim = conv_channels * n_preview_points
        self.preview_branch = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Combine and project to features_dim
        self.combine = nn.Sequential(
            nn.Linear(state_hidden_dim + conv_flat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state   = obs[:, : self._n_state]
        preview = obs[:, self._n_state :].unsqueeze(1)  # (B, 1, n_preview)

        state_feat   = self.state_branch(state)
        preview_feat = self.preview_branch(preview)

        return self.combine(torch.cat([state_feat, preview_feat], dim=1))

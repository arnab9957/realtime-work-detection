# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Regression head producing tracking features for cross-frame re-identification."""

import os
import torch
import torch.nn as nn

TRACKFEAT_DIMS = {
    'sam2.1_hiera_large__pooler_avg8': 4096,
}

class TrackFeatRegressor(nn.Module):
    
    def __init__(self, trackfeat_name, input_dim, trackfeat_arch):
        """Build the tracking feature regression head.

        Args:
            trackfeat_name: Name of the target tracking feature space; must be a
                key in TRACKFEAT_DIMS.
            input_dim: Dimension of the input feature vector.
            trackfeat_arch: Architecture type; currently only "MLp" is supported,
                which builds a two-layer MLP with a ReLU activation.
        """
        super().__init__()
        self.trackfeat_name = trackfeat_name
        self.input_dim = input_dim
        self.trackfeat_arch = trackfeat_arch
        trackfeat_dim = TRACKFEAT_DIMS[trackfeat_name]
        assert trackfeat_arch=="MLp", "not implemented otherwise"
        self.trackfeat_model = nn.Sequential(*[nn.Linear(input_dim, trackfeat_dim), nn.ReLU(), nn.Linear(trackfeat_dim,trackfeat_dim)])
    
    def forward(self, x):
        """Project input features to the tracking feature space.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Tensor of shape (..., trackfeat_dim).
        """
        return self.trackfeat_model(x)

# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Tensor manipulation helpers (rotation matrix utilities)."""

import torch

def rotation_to_homogeneous(x):
    """Embed a batch of 3x3 rotation matrices into 4x4 homogeneous matrices.

    The translation component is set to zero and the bottom row to [0, 0, 0, 1].

    Args:
        x: Rotation matrix tensor of shape (..., 3, 3).

    Returns:
        Homogeneous matrix tensor of shape (..., 4, 4).
    """
    return torch.cat([torch.cat([x, torch.zeros(x.shape[:-2] + (3, 1), device=x.device)], dim=-1),
                      torch.tensor([0, 0, 0, 1], dtype=x.dtype, device=x.device).expand(x.shape[:-2] + (1, 4))], dim=-2)
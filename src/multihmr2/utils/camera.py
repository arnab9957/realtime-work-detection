# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Camera projection helpers (perspective and inverse-perspective)."""

import torch
from torch import Tensor


def perspective_projection(x, K):
    """
    This function computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """
    # Apply perspective distortion
    y = x / x[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Apply camera intrinsics
    y = torch.einsum("bij,bkj->bki", K, y)  # (bs, N, 3)

    return y[:, :, :2]


def inverse_perspective_projection(points, K, distance: Tensor | None = None):
    """
    This function computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (*, N, 2): 2D points
        K (*,3,3): camera intrinsics params
        distance (*, N, 1): distance in the 3D world
    Similar to:
        - pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    """
    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)

    K_inv = torch.linalg.inv(K)
    points = torch.matmul(points.unsqueeze(-2), K_inv.transpose(-1, -2)).squeeze(-2)
    # Apply perspective distortion
    return points if distance is None else points * distance

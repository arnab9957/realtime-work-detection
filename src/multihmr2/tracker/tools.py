# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Tracking utilities: Hungarian matching, ridge regression, quaternion helpers."""

import numpy as np
from sklearn.linear_model import Ridge as SkRidge
from scipy.optimize import linear_sum_assignment
import torch
import roma

######## utilitaries

def get_combine_function(combine_str):
    """Return a function that combines feature and pelvis similarity scores.

    Only 'wsum_<wf>_<wp>' format is supported: computes a normalized weighted sum
    (wf/(wf+wp))*feat_sim + (wp/(wf+wp))*pelvis_sim.

    Args:
        combine_str: String encoding the combination method and weights.

    Returns:
        A callable that takes (feat_sim, pelvis_sim) and returns a combined
        similarity tensor.

    Raises:
        ValueError: If combine_str is not a recognized format.
    """
    if combine_str.startswith('wsum'): # same without the bug
        w_feat = float(combine_str.split('_')[1])
        w_pelvis = float(combine_str.split('_')[2])
        weight_sum = w_feat + w_pelvis
        w_feat /= weight_sum
        w_pelvis /= weight_sum   
        combine_fun = lambda feat_sim, pelvis_sim: (feat_sim*w_feat) + (pelvis_sim*w_pelvis)
    else:
        raise ValueError(f'Unknown combination function {combine_str}')
    return combine_fun
    
######## for rotation

def toquat(rotmat):
    """Convert rotation matrices to canonical unit quaternions (w >= 0).

    Uses roma to convert to unit quaternions, then flips the sign of any quaternion
    whose w component is negative, ensuring a canonical positive-w representation.

    Args:
        rotmat: Rotation matrix tensor of shape (..., 3, 3).

    Returns:
        Unit quaternion tensor of shape (..., 4) in (x, y, z, w) order with w >= 0.
    """
    quat = roma.rotmat_to_unitquat(rotmat)
    sign = torch.sign(quat[..., 3:4]) 
    sign[sign == 0] = 1.0  # if exactly zero, keep as-is
    return quat * sign

def quat_normalize(q, eps=1e-12):
    """Normalize quaternions to unit length."""
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

def quat_batched_cdist_geodesic_deg(A, B):
    """
    Batched pairwise geodesic distance (radians).
    A: (B,N,4)
    B: (B,M,4)
    Returns: (B,N,M)
    """
    A = quat_normalize(A)
    B = quat_normalize(B)
    # einsum over quaternion dimension
    dots = torch.abs(torch.einsum('bnk,bmk->bnm', A, B))
    dots = dots.clamp(-1.0, 1.0)
    angles = 2.0 * torch.acos(dots)
    return angles

######## Hungarian Matching

really_large_value = 9999999999999
def hungarian_matching(cost_matrix: torch.Tensor, max_threshold: float, max_val=really_large_value):
    """
    Performs Hungarian Matching with an individual element cost threshold.

    Args:
        cost_matrix (torch.Tensor): A rectangular or square cost matrix (m x n).
        max_threshold (float): Maximum allowed cost for any assignment.
        max_val: Cost value used to disable assignments above max_threshold.

    Returns:
        row_indices (torch.Tensor): Indices of assigned rows.
        col_indices (torch.Tensor): Indices of assigned columns.
    """
    device = cost_matrix.device
    cost_matrix_np = cost_matrix.cpu().numpy()

    # Handle non-square matrices by padding with a large value (inf)
    m, n = cost_matrix_np.shape
    max_dim = max(m, n)
    padded_cost_matrix = torch.full((max_dim, max_dim), really_large_value, dtype=cost_matrix.dtype, device=device)
    padded_cost_matrix[:m, :n] = cost_matrix  # Copy original values

    # Apply threshold: Assignments exceeding max_threshold are disabled by setting them to inf
    if max_val is not None: padded_cost_matrix[padded_cost_matrix > max_threshold] = max_val
    
    # Convert to numpy for the algorithm
    cost_matrix_np = padded_cost_matrix.cpu().numpy()
    
    # Apply Hungarian Algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix_np)

    # Convert results back to PyTorch tensors
    row_indices = torch.tensor(row_indices, dtype=torch.long, device=device)
    col_indices = torch.tensor(col_indices, dtype=torch.long, device=device)

    # Filter out invalid assignments due to thresholding (inf values)
    valid_mask = (row_indices < m) & (col_indices < n) & (padded_cost_matrix[row_indices, col_indices] <= max_threshold)
    row_indices = row_indices[valid_mask]
    col_indices = col_indices[valid_mask]

    return row_indices, col_indices

######### Ridge Regression

class Ridge(object):
    def __init__(self, alpha = 0, fit_intercept = True,):
        """Initialize ridge regression with optional intercept.

        Args:
            alpha: Regularization strength (L2 penalty). 0 means no regularization.
            fit_intercept: If True, an unregularized bias term is added to the model.
        """
        self.alpha = alpha
        self.sqrt_alpha = np.sqrt(self.alpha)
        self.fit_intercept = fit_intercept
        
    def fit(self, X: torch.tensor, Y: torch.tensor) -> None:
        """Fit the ridge regression model using least squares on augmented data.

        Regularization is applied by appending scaled identity rows to the design
        matrix. If fit_intercept is True, the intercept column is excluded from
        regularization. Solves the system with torch.linalg.lstsq.

        Args:
            X: Input tensor of shape (n_samples, n_features).
            Y: Target tensor of shape (n_samples, n_targets).
        """
        assert X.ndim==2 and Y.ndim==2
        assert X.shape[0] == Y.shape[0], "Number of X and y rows don't match"
        device, dtype = X.device, X.dtype
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=device, dtype=dtype), X], dim = 1)
        # Identity matrix that excludes intercept from regularization
        reg_mask = torch.cat([
            torch.zeros(1, device=device, dtype=dtype) if self.fit_intercept else torch.tensor([], device=device, dtype=dtype),
            torch.ones(X.shape[1] - 1 if self.fit_intercept else X.shape[1], device=device, dtype=dtype),
        ])
        I = torch.diag(reg_mask) * self.sqrt_alpha

        # Augment
        X_aug = torch.cat([X, I], dim=0)
        Y_aug = torch.cat([Y, torch.zeros((X.shape[1], Y.shape[1]), device=device, dtype=dtype)], dim=0)

        # Use lstsq to solve
        self.w = torch.linalg.lstsq(X_aug, Y_aug).solution



    def predict(self, X: torch.tensor):
        """Predict targets for the given input.

        Args:
            X: Input tensor of shape (n_samples, n_features).

        Returns:
            Predicted target tensor of shape (n_samples, n_targets).
        """
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype), X], dim = 1)
        return X @ self.w
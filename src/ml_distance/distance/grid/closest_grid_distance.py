"""
BAS-HMR: Closest 3D grid-point distance between two Object3DGrid objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .object_grid import Object3DGrid


@dataclass
class ClosestGridPair:
    distance_m: float
    first_point_xyz: Tuple[float, float, float]
    second_point_xyz: Tuple[float, float, float]
    first_index: int
    second_index: int


def closest_grid_point_distance(
    first_grid: Object3DGrid,
    second_grid: Object3DGrid,
) -> Optional[ClosestGridPair]:
    """
    Compare every valid 3D point in the first grid against every
    valid 3D point in the second grid.

    Returns the closest pair.
    """

    first_xyz = np.asarray(
        first_grid.xyz_array(),
        dtype=np.float32,
    )

    second_xyz = np.asarray(
        second_grid.xyz_array(),
        dtype=np.float32,
    )

    if first_xyz.ndim != 2 or first_xyz.shape[1] != 3:
        raise ValueError(
            f"Invalid first grid shape: {first_xyz.shape}"
        )

    if second_xyz.ndim != 2 or second_xyz.shape[1] != 3:
        raise ValueError(
            f"Invalid second grid shape: {second_xyz.shape}"
        )

    if len(first_xyz) == 0 or len(second_xyz) == 0:
        return None

    # --------------------------------------------------------
    # Pairwise Euclidean distances
    #
    # first_xyz  -> (N, 3)
    # second_xyz -> (M, 3)
    #
    # result     -> (N, M)
    # --------------------------------------------------------

    delta = (
        first_xyz[:, None, :]
        - second_xyz[None, :, :]
    )

    distances = np.linalg.norm(
        delta,
        axis=2,
    )

    flat_index = int(np.argmin(distances))

    first_index, second_index = np.unravel_index(
        flat_index,
        distances.shape,
    )

    return ClosestGridPair(
        distance_m=float(
            distances[first_index, second_index]
        ),
        first_point_xyz=tuple(
            float(v)
            for v in first_xyz[first_index]
        ),
        second_point_xyz=tuple(
            float(v)
            for v in second_xyz[second_index]
        ),
        first_index=int(first_index),
        second_index=int(second_index),
    )

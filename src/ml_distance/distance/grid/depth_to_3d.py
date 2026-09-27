"""
BAS-HMR
Depth Map -> 3D Camera Coordinate Conversion

Converts pixel coordinates and metric depth into camera-frame
3D coordinates.

Coordinate system:

        Camera
          |
          | Z
          |
          +-------- X
         /
        Y

Formula:

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    These are development values initially.
    They should eventually be replaced by measured/calibrated
    intrinsics for the final system.
    """

    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0


@dataclass
class Point3D:
    """Single 3D point in metres."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z],
            dtype=np.float32,
        )

    def distance_to(self, other: "Point3D") -> float:
        """Euclidean 3D distance in metres."""

        return float(
            np.linalg.norm(
                self.as_array() - other.as_array()
            )
        )

    def ground_distance_to(self, other: "Point3D") -> float:
        """
        Distance on the X-Z ground plane.

        Y is ignored.
        """

        dx = self.x - other.x
        dz = self.z - other.z

        return float(np.sqrt(dx * dx + dz * dz))


class DepthTo3D:
    """
    Converts a metric depth map into camera-frame 3D coordinates.
    """

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        min_depth: float = 0.05,
        max_depth: float = 20.0,
    ):
        self.intrinsics = intrinsics or CameraIntrinsics()

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth: float,
    ) -> Optional[Point3D]:
        """
        Convert one pixel + depth value into 3D.

        Parameters
        ----------
        u : pixel x-coordinate
        v : pixel y-coordinate
        depth : depth in metres
        """

        if not np.isfinite(depth):
            return None

        if depth < self.min_depth or depth > self.max_depth:
            return None

        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        if fx <= 0 or fy <= 0:
            raise ValueError("Invalid focal length.")

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return Point3D(
            x=float(x),
            y=float(y),
            z=float(z),
        )

    def depth_map_to_3d(
        self,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        """
        Convert an entire depth map to a 3D array.

        Returns
        -------
        np.ndarray
            Shape: (H, W, 3)

            [..., 0] = X
            [..., 1] = Y
            [..., 2] = Z
        """

        if depth_map.ndim != 2:
            raise ValueError(
                "depth_map must be a 2D array."
            )

        height, width = depth_map.shape

        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        u = np.arange(
            width,
            dtype=np.float32,
        )

        v = np.arange(
            height,
            dtype=np.float32,
        )

        uu, vv = np.meshgrid(u, v)

        z = depth_map.astype(
            np.float32,
            copy=False,
        )

        valid = (
            np.isfinite(z)
            & (z >= self.min_depth)
            & (z <= self.max_depth)
        )

        x = np.zeros_like(z)
        y = np.zeros_like(z)

        x[valid] = (
            (uu[valid] - cx)
            * z[valid]
            / fx
        )

        y[valid] = (
            (vv[valid] - cy)
            * z[valid]
            / fy
        )

        result = np.stack(
            [x, y, z],
            axis=-1,
        )

        result[~valid] = np.nan

        return result


if __name__ == "__main__":

    print("=" * 60)
    print("DEPTH -> 3D SELF TEST")
    print("=" * 60)

    intrinsics = CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )

    converter = DepthTo3D(intrinsics)

    point = converter.pixel_to_3d(
        u=320,
        v=240,
        depth=2.0,
    )

    print(
        f"Center pixel -> "
        f"X={point.x:.3f}m "
        f"Y={point.y:.3f}m "
        f"Z={point.z:.3f}m"
    )

    assert point is not None
    assert abs(point.x) < 1e-6
    assert abs(point.y) < 1e-6
    assert abs(point.z - 2.0) < 1e-6

    point2 = converter.pixel_to_3d(
        u=570,
        v=240,
        depth=2.0,
    )

    distance = point.distance_to(point2)

    print(
        f"Second point -> "
        f"X={point2.x:.3f}m "
        f"Y={point2.y:.3f}m "
        f"Z={point2.z:.3f}m"
    )

    print(
        f"3D distance = {distance:.3f}m"
    )

    assert abs(distance - 1.0) < 1e-5

    print()
    print("DEPTH -> 3D SELF TEST PASSED")

"""
BAS-HMR
Object-Centric 3D Grid

Extracts a fixed-size grid of 3D points from an object's
bounding box and a metric depth map.

Pipeline:

    Bounding Box
         +
      Depth Map
         ↓
       5 x 5
      sampling
         ↓
    25 3D points
         ↓
       X, Y, Z

Distances remain in metres internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .depth_to_3d import DepthTo3D, Point3D


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass
class BoundingBox:
    """2D object bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


# ============================================================
# GRID POINT
# ============================================================

@dataclass
class GridPoint:
    """One valid sampled point in the object grid."""

    row: int
    col: int

    u: float
    v: float

    point_3d: Point3D

    depth: float


# ============================================================
# OBJECT 3D GRID
# ============================================================

@dataclass
class Object3DGrid:
    """
    Complete 3D representation of one detected object.

    IMPORTANT:
    This object is intentionally NOT a dictionary.

    FeatureExtractor expects this class and calls:
        grid.xyz_array()
        grid.depth_array()
    """

    object_id: int
    object_class: str

    bbox: BoundingBox

    grid_rows: int
    grid_cols: int

    points: list[GridPoint]

    valid_ratio: float

    def xyz_array(self) -> np.ndarray:
        """
        Return all valid XYZ points.

        Returns
        -------
        np.ndarray
            Shape (N, 3), where columns are X, Y, Z in metres.
        """

        if not self.points:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(
            [
                point.point_3d.as_array()
                for point in self.points
            ],
            dtype=np.float32,
        )

    def depth_array(self) -> np.ndarray:
        """Return all valid sampled depths in metres."""

        if not self.points:
            return np.empty((0,), dtype=np.float32)

        return np.asarray(
            [point.depth for point in self.points],
            dtype=np.float32,
        )

    @property
    def point_count(self) -> int:
        """Number of valid 3D points."""

        return len(self.points)

    @property
    def centroid(self) -> Optional[Point3D]:
        """Return the centroid of all valid 3D points."""

        xyz = self.xyz_array()

        if xyz.size == 0:
            return None

        return Point3D(
            x=float(np.mean(xyz[:, 0])),
            y=float(np.mean(xyz[:, 1])),
            z=float(np.mean(xyz[:, 2])),
        )


# ============================================================
# OBJECT GRID EXTRACTOR
# ============================================================

class ObjectGridExtractor:
    """
    Extracts a fixed-size 3D grid from a detected object.

    Parameters
    ----------
    depth_converter:
        DepthTo3D instance.

    rows:
        Number of vertical grid samples.

    cols:
        Number of horizontal grid samples.

    border_ratio:
        Fraction of the bounding box ignored around the edges.

    depth_radius:
        Radius of neighborhood used for robust median depth.
    """

    def __init__(
        self,
        depth_converter: Optional[DepthTo3D] = None,
        rows: int = 5,
        cols: int = 5,
        border_ratio: float = 0.10,
        depth_radius: int = 2,
    ):
        if rows < 2:
            raise ValueError("rows must be >= 2")

        if cols < 2:
            raise ValueError("cols must be >= 2")

        if not 0.0 <= border_ratio < 0.5:
            raise ValueError(
                "border_ratio must be between 0 and 0.5"
            )

        if depth_radius < 0:
            raise ValueError("depth_radius must be >= 0")

        self.depth_converter = depth_converter or DepthTo3D()

        self.rows = int(rows)
        self.cols = int(cols)
        self.border_ratio = float(border_ratio)
        self.depth_radius = int(depth_radius)

    # --------------------------------------------------------
    # BBOX NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize_bbox(bbox) -> BoundingBox:
        """
        Accept BoundingBox as the canonical type.

        Also accepts a dictionary or a 4-value sequence so that
        the grid module is robust when called from detection code.
        """

        if isinstance(bbox, BoundingBox):
            values = (
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
            )

        elif isinstance(bbox, dict):
            try:
                values = (
                    bbox["x1"],
                    bbox["y1"],
                    bbox["x2"],
                    bbox["y2"],
                )
            except KeyError as exc:
                raise ValueError(
                    "Bounding box dictionary must contain "
                    "x1, y1, x2, y2."
                ) from exc

        elif hasattr(bbox, "x1") and hasattr(bbox, "y1")                 and hasattr(bbox, "x2") and hasattr(bbox, "y2"):
            values = (
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
            )

        else:
            arr = np.asarray(bbox, dtype=np.float32).reshape(-1)

            if arr.size != 4:
                raise TypeError(
                    "bbox must be BoundingBox, dict, or "
                    "a 4-value sequence."
                )

            values = tuple(arr.tolist())

        try:
            x1, y1, x2, y2 = map(float, values)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Bounding box coordinates must be numeric."
            ) from exc

        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError(
                "Bounding box contains NaN or infinite values."
            )

        # Make coordinate ordering robust.
        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        return BoundingBox(x1, y1, x2, y2)

    # --------------------------------------------------------
    # BBOX CLIPPING
    # --------------------------------------------------------

    @staticmethod
    def _clip_bbox(
        bbox: BoundingBox,
        width: int,
        height: int,
    ) -> BoundingBox:
        """Clip a bounding box safely to image boundaries."""

        if width <= 0 or height <= 0:
            raise ValueError(
                "Image width and height must be positive."
            )

        return BoundingBox(
            x1=float(np.clip(bbox.x1, 0, width - 1)),
            y1=float(np.clip(bbox.y1, 0, height - 1)),
            x2=float(np.clip(bbox.x2, 0, width - 1)),
            y2=float(np.clip(bbox.y2, 0, height - 1)),
        )

    # --------------------------------------------------------
    # DEPTH SAMPLING
    # --------------------------------------------------------

    def _sample_depth(
        self,
        depth_map: np.ndarray,
        u: float,
        v: float,
    ) -> Optional[float]:
        """
        Obtain a robust depth value around one image coordinate.

        A local median is used instead of trusting a single pixel.
        """

        height, width = depth_map.shape

        x = int(round(u))
        y = int(round(v))

        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))

        radius = self.depth_radius

        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)

        region = depth_map[y1:y2, x1:x2]

        valid = region[
            np.isfinite(region)
            & (
                region >= self.depth_converter.min_depth
            )
            & (
                region <= self.depth_converter.max_depth
            )
        ]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    def extract(
        self,
        depth_map: np.ndarray,
        bbox: BoundingBox,
        object_id: int = 0,
        object_class: str = "object",
    ) -> Optional[Object3DGrid]:
        """
        Extract a fixed-size 3D grid.

        IMPORTANT API CONTRACT
        ----------------------
        depth_map comes FIRST.
        bbox comes SECOND.

        Returns
        -------
        Object3DGrid or None

        Never returns a dictionary.
        """

        # ----------------------------------------------------
        # Validate depth map
        # ----------------------------------------------------

        if depth_map is None:
            return None

        depth_map = np.asarray(depth_map, dtype=np.float32)

        if depth_map.ndim != 2:
            raise ValueError(
                f"depth_map must be 2D, got shape {depth_map.shape}."
            )

        height, width = depth_map.shape

        if height < 2 or width < 2:
            return None

        # ----------------------------------------------------
        # Normalize bbox
        # ----------------------------------------------------

        bbox = self._normalize_bbox(bbox)

        bbox = self._clip_bbox(
            bbox,
            width,
            height,
        )

        if bbox.width < 2 or bbox.height < 2:
            return None

        # ----------------------------------------------------
        # Ignore small boundary region
        # ----------------------------------------------------

        margin_x = bbox.width * self.border_ratio
        margin_y = bbox.height * self.border_ratio

        left = bbox.x1 + margin_x
        right = bbox.x2 - margin_x

        top = bbox.y1 + margin_y
        bottom = bbox.y2 - margin_y

        if right <= left or bottom <= top:
            return None

        # ----------------------------------------------------
        # Sample fixed grid
        # ----------------------------------------------------

        points: list[GridPoint] = []

        total_points = self.rows * self.cols

        for row in range(self.rows):

            if self.rows == 1:
                v = (top + bottom) / 2.0
            else:
                v = (
                    top
                    + row * (bottom - top)
                    / (self.rows - 1)
                )

            for col in range(self.cols):

                if self.cols == 1:
                    u = (left + right) / 2.0
                else:
                    u = (
                        left
                        + col * (right - left)
                        / (self.cols - 1)
                    )

                depth = self._sample_depth(
                    depth_map,
                    u,
                    v,
                )

                if depth is None:
                    continue

                point_3d = self.depth_converter.pixel_to_3d(
                    u,
                    v,
                    depth,
                )

                if point_3d is None:
                    continue

                xyz = np.asarray(
                    point_3d.as_array(),
                    dtype=np.float32,
                ).reshape(-1)

                if xyz.size != 3:
                    continue

                if not np.all(np.isfinite(xyz)):
                    continue

                points.append(
                    GridPoint(
                        row=row,
                        col=col,
                        u=float(u),
                        v=float(v),
                        point_3d=point_3d,
                        depth=float(depth),
                    )
                )

        # ----------------------------------------------------
        # Valid ratio
        # ----------------------------------------------------

        valid_ratio = (
            len(points) / total_points
            if total_points > 0
            else 0.0
        )

        if not points:
            return None

        # ----------------------------------------------------
        # IMPORTANT:
        # Return Object3DGrid, never dict.
        # ----------------------------------------------------

        return Object3DGrid(
            object_id=int(object_id),
            object_class=str(object_class),
            bbox=bbox,
            grid_rows=self.rows,
            grid_cols=self.cols,
            points=points,
            valid_ratio=float(valid_ratio),
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 64)
    print("OBJECT 3D GRID SELF TEST")
    print("=" * 64)

    height = 480
    width = 640

    # Synthetic metric depth map.
    depth_map = np.full(
        (height, width),
        2.0,
        dtype=np.float32,
    )

    # Use the project's existing DepthTo3D API.
    # Camera intrinsics are owned by DepthTo3D/CameraIntrinsics,
    # so do not pass fx/fy/cx/cy directly to DepthTo3D.
    converter = DepthTo3D(
        min_depth=0.05,
        max_depth=20.0,
    )

    bbox = BoundingBox(
        x1=120,
        y1=100,
        x2=520,
        y2=400,
    )

    extractor = ObjectGridExtractor(
        depth_converter=converter,
        rows=5,
        cols=5,
        border_ratio=0.10,
        depth_radius=2,
    )

    grid = extractor.extract(
        depth_map=depth_map,
        bbox=bbox,
        object_id=1,
        object_class="person",
    )

    assert grid is not None
    assert isinstance(grid, Object3DGrid)
    assert hasattr(grid, "xyz_array")
    assert hasattr(grid, "depth_array")

    print(f"Object ID       : {grid.object_id}")
    print(f"Object class    : {grid.object_class}")
    print(
        f"Grid size       : "
        f"{grid.grid_rows} x {grid.grid_cols}"
    )
    print(f"Valid points    : {len(grid.points)}")
    print(f"Valid ratio     : {grid.valid_ratio:.2%}")

    xyz = grid.xyz_array()

    print(f"XYZ shape       : {xyz.shape}")

    assert xyz.shape == (25, 3)
    assert len(grid.points) == 25

    print()
    print("Sample 3D points:")

    for point in grid.points[:5]:
        print(
            f"  ({point.u:.1f}, {point.v:.1f}) "
            f"-> "
            f"X={point.point_3d.x:.3f} "
            f"Y={point.point_3d.y:.3f} "
            f"Z={point.point_3d.z:.3f}"
        )

    assert abs(
        np.median(xyz[:, 2]) - 2.0
    ) < 1e-6

    # --------------------------------------------------------
    # Dictionary bbox compatibility test
    # --------------------------------------------------------

    dict_bbox = {
        "x1": 120,
        "y1": 100,
        "x2": 520,
        "y2": 400,
    }

    grid_dict_bbox = extractor.extract(
        depth_map=depth_map,
        bbox=dict_bbox,
        object_id=2,
        object_class="chair",
    )

    assert grid_dict_bbox is not None
    assert isinstance(grid_dict_bbox, Object3DGrid)
    assert grid_dict_bbox.object_class == "chair"
    assert grid_dict_bbox.xyz_array().shape == (25, 3)

    print()
    print("Dictionary bbox compatibility test PASSED")

    # --------------------------------------------------------
    # Invalid-depth test
    # --------------------------------------------------------

    bad_depth = depth_map.copy()
    bad_depth[:100, :] = np.nan
    bad_depth[100:150, :] = 999.0

    grid_bad = extractor.extract(
        depth_map=bad_depth,
        bbox=bbox,
        object_id=3,
        object_class="person",
    )

    assert grid_bad is None or isinstance(
        grid_bad,
        Object3DGrid,
    )

    print("Invalid depth handling test PASSED")

    print()
    print("=" * 64)
    print("OBJECT 3D GRID SELF TEST PASSED")
    print("=" * 64)

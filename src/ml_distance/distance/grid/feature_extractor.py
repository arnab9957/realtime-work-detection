"""
BAS-HMR
3D Object Feature Extractor

Converts an Object3DGrid into a compact numerical
feature vector suitable for machine-learning models.

Input:
    Object3DGrid
        ↓
    25 XYZ points

Output:
    numerical feature dictionary / vector
"""

from typing import Dict

import numpy as np

from .object_grid import (
    Object3DGrid,
)


class FeatureExtractor:
    """
    Extract statistical and geometric features from
    an object-centric 3D grid.
    """

    def __init__(self):
        pass

    def extract(
        self,
        grid: Object3DGrid,
    ) -> Dict[str, float]:
        """
        Extract features from one object's 3D grid.
        """

        xyz = grid.xyz_array()

        if xyz.size == 0:
            raise ValueError(
                "Object3DGrid contains no valid 3D points."
            )

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        features: Dict[str, float] = {}

        # -------------------------------------------------
        # Basic information
        # -------------------------------------------------

        features["object_id"] = float(
            grid.object_id
        )

        features["valid_ratio"] = float(
            grid.valid_ratio
        )

        features["valid_points"] = float(
            len(grid.points)
        )

        # -------------------------------------------------
        # 3D centroid
        # -------------------------------------------------

        features["centroid_x"] = float(
            np.mean(x)
        )

        features["centroid_y"] = float(
            np.mean(y)
        )

        features["centroid_z"] = float(
            np.mean(z)
        )

        # -------------------------------------------------
        # Median position
        # -------------------------------------------------

        features["median_x"] = float(
            np.median(x)
        )

        features["median_y"] = float(
            np.median(y)
        )

        features["median_z"] = float(
            np.median(z)
        )

        # -------------------------------------------------
        # Minimum coordinates
        # -------------------------------------------------

        features["min_x"] = float(
            np.min(x)
        )

        features["min_y"] = float(
            np.min(y)
        )

        features["min_z"] = float(
            np.min(z)
        )

        # -------------------------------------------------
        # Maximum coordinates
        # -------------------------------------------------

        features["max_x"] = float(
            np.max(x)
        )

        features["max_y"] = float(
            np.max(y)
        )

        features["max_z"] = float(
            np.max(z)
        )

        # -------------------------------------------------
        # 3D spatial dimensions
        # -------------------------------------------------

        features["width_3d"] = float(
            np.max(x) - np.min(x)
        )

        features["height_3d"] = float(
            np.max(y) - np.min(y)
        )

        features["depth_3d"] = float(
            np.max(z) - np.min(z)
        )

        # -------------------------------------------------
        # Standard deviation
        # -------------------------------------------------

        features["std_x"] = float(
            np.std(x)
        )

        features["std_y"] = float(
            np.std(y)
        )

        features["std_z"] = float(
            np.std(z)
        )

        # -------------------------------------------------
        # Median depth
        # -------------------------------------------------

        features["median_depth"] = float(
            np.median(z)
        )

        # -------------------------------------------------
        # Depth statistics
        # -------------------------------------------------

        features["mean_depth"] = float(
            np.mean(z)
        )

        features["min_depth"] = float(
            np.min(z)
        )

        features["max_depth"] = float(
            np.max(z)
        )

        features["depth_std"] = float(
            np.std(z)
        )

        # -------------------------------------------------
        # Distance of object centroid from camera
        # -------------------------------------------------

        centroid = np.array(
            [
                features["centroid_x"],
                features["centroid_y"],
                features["centroid_z"],
            ],
            dtype=np.float32,
        )

        features["centroid_distance"] = float(
            np.linalg.norm(centroid)
        )

        # -------------------------------------------------
        # Horizontal ground distance from camera
        # -------------------------------------------------

        features["centroid_ground_distance"] = float(
            np.sqrt(
                centroid[0] ** 2
                + centroid[2] ** 2
            )
        )

        return features

    @staticmethod
    def feature_names():
        """
        Return feature names in deterministic order.
        """

        return [
            "object_id",
            "valid_ratio",
            "valid_points",

            "centroid_x",
            "centroid_y",
            "centroid_z",

            "median_x",
            "median_y",
            "median_z",

            "min_x",
            "min_y",
            "min_z",

            "max_x",
            "max_y",
            "max_z",

            "width_3d",
            "height_3d",
            "depth_3d",

            "std_x",
            "std_y",
            "std_z",

            "median_depth",
            "mean_depth",
            "min_depth",
            "max_depth",
            "depth_std",

            "centroid_distance",
            "centroid_ground_distance",
        ]

    def to_vector(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:
        """
        Convert feature dictionary into a deterministic
        NumPy feature vector.
        """

        names = self.feature_names()

        return np.asarray(
            [
                features[name]
                for name in names
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":

    from .depth_to_3d import DepthTo3D
    from .object_grid import (
        BoundingBox,
        ObjectGridExtractor,
    )

    print("=" * 60)
    print("3D FEATURE EXTRACTOR SELF TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Synthetic depth map
    # -------------------------------------------------

    height = 480
    width = 640

    depth_map = np.full(
        (height, width),
        2.0,
        dtype=np.float32,
    )

    # -------------------------------------------------
    # 3D converter
    # -------------------------------------------------

    converter = DepthTo3D()

    # -------------------------------------------------
    # Object bounding box
    # -------------------------------------------------

    bbox = BoundingBox(
        x1=120,
        y1=100,
        x2=520,
        y2=400,
    )

    # -------------------------------------------------
    # Grid
    # -------------------------------------------------

    grid_extractor = ObjectGridExtractor(
        depth_converter=converter,
        rows=5,
        cols=5,
        border_ratio=0.10,
        depth_radius=2,
    )

    grid = grid_extractor.extract(
        depth_map=depth_map,
        bbox=bbox,
        object_id=1,
        object_class="person",
    )

    assert grid is not None

    # -------------------------------------------------
    # Feature extraction
    # -------------------------------------------------

    extractor = FeatureExtractor()

    features = extractor.extract(
        grid
    )

    vector = extractor.to_vector(
        features
    )

    # -------------------------------------------------
    # Display
    # -------------------------------------------------

    print(
        f"Number of features : {len(features)}"
    )

    print(
        f"Vector shape       : {vector.shape}"
    )

    print()
    print("Important features:")
    print(
        f"Centroid X         : "
        f"{features['centroid_x']:.4f} m"
    )

    print(
        f"Centroid Y         : "
        f"{features['centroid_y']:.4f} m"
    )

    print(
        f"Centroid Z         : "
        f"{features['centroid_z']:.4f} m"
    )

    print(
        f"3D width           : "
        f"{features['width_3d']:.4f} m"
    )

    print(
        f"3D height          : "
        f"{features['height_3d']:.4f} m"
    )

    print(
        f"3D depth           : "
        f"{features['depth_3d']:.4f} m"
    )

    print(
        f"Centroid distance  : "
        f"{features['centroid_distance']:.4f} m"
    )

    print(
        f"Ground distance    : "
        f"{features['centroid_ground_distance']:.4f} m"
    )

    print()
    print("Feature vector:")
    print(vector)

    # -------------------------------------------------
    # Assertions
    # -------------------------------------------------

    assert len(features) == 28

    assert vector.shape == (28,)

    assert abs(
        features["centroid_z"] - 2.0
    ) < 1e-6

    assert (
        features["valid_points"] == 25
    )

    assert (
        features["valid_ratio"] == 1.0
    )

    print()
    print(
        "3D FEATURE EXTRACTOR SELF TEST PASSED"
    )

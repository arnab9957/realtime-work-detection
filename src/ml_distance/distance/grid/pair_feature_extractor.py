"""
BAS-HMR
Person-Chair Pair Feature Extractor

Combines the 3D features of a person and a chair
into a pairwise feature vector.

The output is designed for later distance regression.

IMPORTANT:
    Ground-truth distance is NOT calculated here.

    Ground truth will come from the real tape-measured
    distance during dataset collection.
"""

from typing import Dict

import numpy as np

from .feature_extractor import FeatureExtractor
from .object_grid import Object3DGrid


class PairFeatureExtractor:
    """
    Extract features describing the spatial relationship
    between one person and one chair.
    """

    def __init__(self):
        self.object_feature_extractor = FeatureExtractor()

    def extract(
        self,
        person_grid: Object3DGrid,
        chair_grid: Object3DGrid,
    ) -> Dict[str, float]:

        person_features = (
            self.object_feature_extractor.extract(
                person_grid
            )
        )

        chair_features = (
            self.object_feature_extractor.extract(
                chair_grid
            )
        )

        features: Dict[str, float] = {}

        # =================================================
        # PERSON FEATURES
        # =================================================

        features["person_centroid_x"] = (
            person_features["centroid_x"]
        )

        features["person_centroid_y"] = (
            person_features["centroid_y"]
        )

        features["person_centroid_z"] = (
            person_features["centroid_z"]
        )

        features["person_width_3d"] = (
            person_features["width_3d"]
        )

        features["person_height_3d"] = (
            person_features["height_3d"]
        )

        features["person_depth_3d"] = (
            person_features["depth_3d"]
        )

        features["person_depth_std"] = (
            person_features["depth_std"]
        )

        features["person_valid_ratio"] = (
            person_features["valid_ratio"]
        )

        # =================================================
        # CHAIR FEATURES
        # =================================================

        features["chair_centroid_x"] = (
            chair_features["centroid_x"]
        )

        features["chair_centroid_y"] = (
            chair_features["centroid_y"]
        )

        features["chair_centroid_z"] = (
            chair_features["centroid_z"]
        )

        features["chair_width_3d"] = (
            chair_features["width_3d"]
        )

        features["chair_height_3d"] = (
            chair_features["height_3d"]
        )

        features["chair_depth_3d"] = (
            chair_features["depth_3d"]
        )

        features["chair_depth_std"] = (
            chair_features["depth_std"]
        )

        features["chair_valid_ratio"] = (
            chair_features["valid_ratio"]
        )

        # =================================================
        # RELATIVE 3D POSITION
        # =================================================

        dx = (
            person_features["centroid_x"]
            - chair_features["centroid_x"]
        )

        dy = (
            person_features["centroid_y"]
            - chair_features["centroid_y"]
        )

        dz = (
            person_features["centroid_z"]
            - chair_features["centroid_z"]
        )

        features["delta_x"] = float(dx)
        features["delta_y"] = float(dy)
        features["delta_z"] = float(dz)

        # =================================================
        # ABSOLUTE RELATIVE POSITION
        # =================================================

        features["abs_delta_x"] = float(
            abs(dx)
        )

        features["abs_delta_y"] = float(
            abs(dy)
        )

        features["abs_delta_z"] = float(
            abs(dz)
        )

        # =================================================
        # 3D DISTANCE
        # =================================================

        distance_3d = float(
            np.sqrt(
                dx ** 2
                + dy ** 2
                + dz ** 2
            )
        )

        features["raw_3d_distance"] = (
            distance_3d
        )

        # =================================================
        # GROUND / HORIZONTAL DISTANCE
        # =================================================

        ground_distance = float(
            np.sqrt(
                dx ** 2
                + dz ** 2
            )
        )

        features["raw_ground_distance"] = (
            ground_distance
        )

        # =================================================
        # VERTICAL DIFFERENCE
        # =================================================

        features["vertical_difference"] = (
            float(abs(dy))
        )

        # =================================================
        # DEPTH RELATIONSHIP
        # =================================================

        features["person_minus_chair_depth"] = (
            float(
                person_features["median_depth"]
                - chair_features["median_depth"]
            )
        )

        features["absolute_depth_difference"] = (
            float(
                abs(
                    person_features["median_depth"]
                    - chair_features["median_depth"]
                )
            )
        )

        # =================================================
        # CENTROID DISTANCE FROM CAMERA
        # =================================================

        features["person_camera_distance"] = (
            person_features["centroid_distance"]
        )

        features["chair_camera_distance"] = (
            chair_features["centroid_distance"]
        )

        # =================================================
        # OBJECT SIZE RELATIONSHIP
        # =================================================

        features["height_difference"] = (
            float(
                person_features["height_3d"]
                - chair_features["height_3d"]
            )
        )

        features["width_difference"] = (
            float(
                person_features["width_3d"]
                - chair_features["width_3d"]
            )
        )

        # =================================================
        # 3D CENTROID VECTOR MAGNITUDES
        # =================================================

        person_position = np.array(
            [
                person_features["centroid_x"],
                person_features["centroid_y"],
                person_features["centroid_z"],
            ],
            dtype=np.float32,
        )

        chair_position = np.array(
            [
                chair_features["centroid_x"],
                chair_features["centroid_y"],
                chair_features["centroid_z"],
            ],
            dtype=np.float32,
        )

        features["person_position_norm"] = (
            float(
                np.linalg.norm(
                    person_position
                )
            )
        )

        features["chair_position_norm"] = (
            float(
                np.linalg.norm(
                    chair_position
                )
            )
        )

        # =================================================
        # DEPTH RATIO
        # =================================================

        person_z = max(
            person_features["centroid_z"],
            1e-6,
        )

        chair_z = max(
            chair_features["centroid_z"],
            1e-6,
        )

        features["depth_ratio"] = float(
            person_z / chair_z
        )

        return features

    @staticmethod
    def feature_names():

        return [

            # Person
            "person_centroid_x",
            "person_centroid_y",
            "person_centroid_z",

            "person_width_3d",
            "person_height_3d",
            "person_depth_3d",

            "person_depth_std",
            "person_valid_ratio",

            # Chair
            "chair_centroid_x",
            "chair_centroid_y",
            "chair_centroid_z",

            "chair_width_3d",
            "chair_height_3d",
            "chair_depth_3d",

            "chair_depth_std",
            "chair_valid_ratio",

            # Relative position
            "delta_x",
            "delta_y",
            "delta_z",

            "abs_delta_x",
            "abs_delta_y",
            "abs_delta_z",

            # Distances
            "raw_3d_distance",
            "raw_ground_distance",

            # Vertical
            "vertical_difference",

            # Depth
            "person_minus_chair_depth",
            "absolute_depth_difference",

            # Camera distances
            "person_camera_distance",
            "chair_camera_distance",

            # Size
            "height_difference",
            "width_difference",

            # Position norms
            "person_position_norm",
            "chair_position_norm",

            # Depth ratio
            "depth_ratio",
        ]

    def to_vector(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:

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
    print("PERSON-CHAIR PAIR FEATURE SELF TEST")
    print("=" * 60)

    # =====================================================
    # Synthetic depth map
    # =====================================================

    height = 480
    width = 640

    depth_map = np.full(
        (height, width),
        2.0,
        dtype=np.float32,
    )

    # =====================================================
    # Converter
    # =====================================================

    converter = DepthTo3D()

    # =====================================================
    # Grid extractor
    # =====================================================

    grid_extractor = ObjectGridExtractor(
        depth_converter=converter,
        rows=5,
        cols=5,
        border_ratio=0.10,
        depth_radius=2,
    )

    # =====================================================
    # Person
    # =====================================================

    person_bbox = BoundingBox(
        x1=80,
        y1=100,
        x2=280,
        y2=400,
    )

    person_grid = grid_extractor.extract(
        depth_map,
        person_bbox,
        object_id=1,
        object_class="person",
    )

    # =====================================================
    # Chair
    # =====================================================

    chair_bbox = BoundingBox(
        x1=360,
        y1=180,
        x2=520,
        y2=380,
    )

    chair_grid = grid_extractor.extract(
        depth_map,
        chair_bbox,
        object_id=2,
        object_class="chair",
    )

    assert person_grid is not None
    assert chair_grid is not None

    # =====================================================
    # Pair features
    # =====================================================

    extractor = PairFeatureExtractor()

    features = extractor.extract(
        person_grid,
        chair_grid,
    )

    vector = extractor.to_vector(
        features
    )

    # =====================================================
    # Display
    # =====================================================

    print(
        f"Number of pair features : "
        f"{len(features)}"
    )

    print(
        f"Feature vector shape    : "
        f"{vector.shape}"
    )

    print()

    print(
        f"Person centroid         : "
        f"({features['person_centroid_x']:.3f}, "
        f"{features['person_centroid_y']:.3f}, "
        f"{features['person_centroid_z']:.3f})"
    )

    print(
        f"Chair centroid          : "
        f"({features['chair_centroid_x']:.3f}, "
        f"{features['chair_centroid_y']:.3f}, "
        f"{features['chair_centroid_z']:.3f})"
    )

    print()

    print(
        f"Delta X                 : "
        f"{features['delta_x']:.3f} m"
    )

    print(
        f"Delta Y                 : "
        f"{features['delta_y']:.3f} m"
    )

    print(
        f"Delta Z                 : "
        f"{features['delta_z']:.3f} m"
    )

    print()

    print(
        f"Raw 3D distance         : "
        f"{features['raw_3d_distance']:.3f} m"
    )

    print(
        f"Raw ground distance     : "
        f"{features['raw_ground_distance']:.3f} m"
    )

    print()

    print(
        "Feature vector:"
    )

    print(vector)

    # =====================================================
    # Assertions
    # =====================================================

    assert len(features) == 34

    assert vector.shape == (34,)

    assert np.all(
        np.isfinite(vector)
    )

    assert (
        features["person_centroid_z"]
        > 0
    )

    assert (
        features["chair_centroid_z"]
        > 0
    )

    print()

    print(
        "PERSON-CHAIR PAIR FEATURE "
        "SELF TEST PASSED"
    )

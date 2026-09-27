"""
BAS HMR
Metric Depth Estimation

Purpose:
    Estimate indoor metric depth from a camera frame.

Pipeline:

    Camera Frame
        ↓
    RGB Conversion
        ↓
    Depth Anything V2 Metric Indoor
        ↓
    Metric Depth Map
        ↓
    Depth values in metres

CPU compatible.
"""

import time

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "depth-anything/"
    "Depth-Anything-V2-Metric-Indoor-Small-hf"
)

DEVICE = -1

CAMERA_DEVICE = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


# ============================================================
# METRIC DEPTH ESTIMATOR
# ============================================================

class MetricDepthEstimator:
    """
    Depth Anything V2 Metric Indoor estimator.

    The estimator receives an RGB image and returns a depth map.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: int = DEVICE,
    ):

        print()
        print("=" * 70)
        print("LOADING METRIC DEPTH MODEL")
        print("=" * 70)

        print(f"Model : {model_name}")
        print("Device: CPU")

        self.model_name = model_name

        self.estimator = pipeline(
            task="depth-estimation",
            model=model_name,
            device=device,
        )

        print("Metric depth model loaded successfully.")

    # ========================================================
    # ESTIMATE
    # ========================================================

    def estimate(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate metric depth.

        Parameters
        ----------
        frame:
            OpenCV BGR image.

        Returns
        -------
        np.ndarray
            Depth map.
        """

        if frame is None:

            raise ValueError(
                "Input frame is None."
            )

        if not isinstance(frame, np.ndarray):

            raise TypeError(
                "Input frame must be a numpy array."
            )

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # NumPy → PIL
        # ----------------------------------------------------

        image = Image.fromarray(
            rgb
        )

        # ----------------------------------------------------
        # Depth estimation
        # ----------------------------------------------------

        result = self.estimator(
            image
        )

        if "predicted_depth" in result:

            depth = result[
                "predicted_depth"
            ]

            if isinstance(
                depth,
                torch.Tensor
            ):

                depth = (
                    depth
                    .detach()
                    .cpu()
                    .numpy()
                )

        elif "depth" in result:

            depth = result[
                "depth"
            ]

            if isinstance(
                depth,
                Image.Image
            ):

                depth = np.asarray(
                    depth,
                    dtype=np.float32,
                )

            else:

                depth = np.asarray(
                    depth,
                    dtype=np.float32,
                )

        else:

            raise RuntimeError(
                "Depth model returned no depth output."
            )

        # ----------------------------------------------------
        # Remove unnecessary dimensions
        # ----------------------------------------------------

        depth = np.squeeze(
            depth
        )

        # ----------------------------------------------------
        # Ensure float32
        # ----------------------------------------------------

        depth = depth.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Resize to camera resolution
        # ----------------------------------------------------

        depth = cv2.resize(
            depth,
            (
                frame.shape[1],
                frame.shape[0],
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        return depth

    # ========================================================
    # DEPTH AT PIXEL
    # ========================================================

    @staticmethod
    def depth_at_pixel(
        depth_map: np.ndarray,
        x: int,
        y: int,
        radius: int = 3,
    ) -> float:
        """
        Get robust depth around a pixel.

        A small neighborhood is used instead of a single pixel
        to reduce depth noise.
        """

        height, width = (
            depth_map.shape[:2]
        )

        x = int(
            np.clip(
                x,
                0,
                width - 1,
            )
        )

        y = int(
            np.clip(
                y,
                0,
                height - 1,
            )
        )

        x1 = max(
            0,
            x - radius,
        )

        y1 = max(
            0,
            y - radius,
        )

        x2 = min(
            width,
            x + radius + 1,
        )

        y2 = min(
            height,
            y + radius + 1,
        )

        crop = depth_map[
            y1:y2,
            x1:x2,
        ]

        valid = crop[
            np.isfinite(crop)
        ]

        if valid.size == 0:

            return float("nan")

        return float(
            np.median(valid)
        )

    # ========================================================
    # OBJECT DEPTH
    # ========================================================

    @staticmethod
    def depth_at_bbox(
        depth_map: np.ndarray,
        bbox,
        crop_ratio: float = 0.60,
    ) -> float:
        """
        Calculate robust depth inside an object's bounding box.

        The central region is used to reduce background contamination.
        """

        x1, y1, x2, y2 = map(
            int,
            bbox,
        )

        height, width = (
            depth_map.shape[:2]
        )

        x1 = max(
            0,
            min(x1, width - 1),
        )

        y1 = max(
            0,
            min(y1, height - 1),
        )

        x2 = max(
            x1 + 1,
            min(x2, width),
        )

        y2 = max(
            y1 + 1,
            min(y2, height),
        )

        box_width = x2 - x1
        box_height = y2 - y1

        margin_x = int(
            box_width
            * (1.0 - crop_ratio)
            / 2.0
        )

        margin_y = int(
            box_height
            * (1.0 - crop_ratio)
            / 2.0
        )

        cx1 = x1 + margin_x
        cy1 = y1 + margin_y

        cx2 = x2 - margin_x
        cy2 = y2 - margin_y

        crop = depth_map[
            cy1:cy2,
            cx1:cx2,
        ]

        valid = crop[
            np.isfinite(crop)
        ]

        if valid.size == 0:

            return float("nan")

        return float(
            np.median(valid)
        )


# ============================================================
# CAMERA TEST
# ============================================================

def camera_test():

    print()
    print("=" * 70)
    print("OPENING CAMERA")
    print("=" * 70)

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera: "
            f"{CAMERA_DEVICE}"
        )

    print("Camera opened.")

    estimator = MetricDepthEstimator()

    frame_count = 0

    try:

        while frame_count < 5:

            success, frame = (
                cap.read()
            )

            if not success:

                print(
                    "Failed to read frame."
                )

                continue

            frame_count += 1

            start = time.time()

            depth = estimator.estimate(
                frame
            )

            elapsed = (
                time.time()
                - start
            )

            print()
            print(
                f"Frame {frame_count}"
            )

            print(
                f"Depth shape: "
                f"{depth.shape}"
            )

            print(
                f"Depth dtype: "
                f"{depth.dtype}"
            )

            print(
                f"Depth min: "
                f"{np.nanmin(depth):.4f}"
            )

            print(
                f"Depth max: "
                f"{np.nanmax(depth):.4f}"
            )

            print(
                f"Depth median: "
                f"{np.nanmedian(depth):.4f}"
            )

            print(
                f"Inference time: "
                f"{elapsed:.2f} s"
            )

    finally:

        cap.release()

        print()
        print(
            "Camera released."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BAS METRIC DEPTH TEST")
    print("=" * 70)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    camera_test()

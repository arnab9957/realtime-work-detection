"""
Scene Object Agent
Open-vocabulary detection of every object in view (YOLOE prompt-free, ~4.6k classes)
and human-object interaction tagging from wrist keypoints / hand boxes.
Runs alongside the protocol detector and never feeds the procedure FSM.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.types import AstronautPose3D, ExperimentObject

# Classes that describe the person, the room or surfaces rather than manipulable objects
_IGNORED_LABELS = {
    "person", "man", "woman", "boy", "girl", "child", "people", "face", "head", "hair",
    "hand", "arm", "finger", "beard", "eye", "mouth", "nose", "ear", "neck", "skin",
    "room", "wall", "floor", "ceiling", "window", "door", "indoor", "office", "home",
    "living room", "bedroom", "kitchen", "lighting", "light", "shadow", "white", "black",
}

_HOLD_COLOR = (255, 0, 255)
_TOUCH_COLOR = (0, 165, 255)
_IDLE_COLOR = (200, 200, 200)


class SceneObjectAgent:
    """Detects all objects in the frame and which of them the human is interacting with."""

    def __init__(
        self,
        model_path: str = "models/yoloe-11s-seg-pf.pt",
        device: Optional[Any] = None,
        conf: float = 0.30,
        interval: int = 2,
        max_objects: int = 25,
    ):
        self.conf = conf
        self.interval = max(1, interval)
        self.max_objects = max_objects
        self.device = device if device is not None else 0
        self.model = None
        self._frame_count = 0
        self._last: List[Dict[str, Any]] = []

        try:
            from ultralytics import YOLOE
            # The prompt-free checkpoint auto-downloads on first use if missing
            self.model = YOLOE(model_path if os.path.exists(model_path) else os.path.basename(model_path))
            print(f"[Scene Object Agent] Open-vocabulary detector engaged: {model_path} ({len(self.model.names)} classes)")
        except Exception as e:
            print(f"[Scene Object Agent] Disabled: {e}")

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def reset(self):
        self._frame_count = 0
        self._last = []

    def process(
        self,
        frame: np.ndarray,
        pose: Optional[AstronautPose3D] = None,
        objects_state: Optional[Dict[str, ExperimentObject]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns scene objects: label, conf, bbox [x1,y1,x2,y2] and interaction (HOLDING/TOUCHING/NEAR/NONE)."""
        if self.model is None:
            return []

        # Re-detect every `interval` frames; interactions are re-evaluated every frame
        if self._frame_count % self.interval == 0 or not self._last:
            self._last = self._detect(frame)
        self._frame_count += 1
        detections = [dict(d) for d in self._last]

        hands = self._hand_points(pose, objects_state)
        for det in detections:
            det["interaction"] = self._interaction(det["bbox"], hands)
        detections.sort(key=lambda d: (d["interaction"] == "NONE", -d["conf"]))
        return detections

    def _detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False, retina_masks=False)
        except Exception as e:
            print(f"[Scene Object Agent] Inference error: {e}")
            return []
        if not results or results[0].boxes is None:
            return []

        h, w = frame.shape[:2]
        frame_area = float(w * h)
        out = []
        for box in results[0].boxes:
            label = str(self.model.names.get(int(box.cls[0].item()), "object"))
            if label.lower() in _IGNORED_LABELS:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            # Boxes covering most of the frame are scene-level labels ("table", "desk room")
            if (x2 - x1) * (y2 - y1) > 0.6 * frame_area:
                continue
            out.append({
                "label": label,
                "conf": round(float(box.conf[0].item()), 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })
        out.sort(key=lambda d: -d["conf"])
        return out[: self.max_objects]

    @staticmethod
    def _hand_points(pose, objects_state) -> List[Tuple[float, float]]:
        points = []
        if pose is not None and pose.keypoints_2d:
            for name in ("left_wrist", "right_wrist", "wrist"):
                kp = pose.keypoints_2d.get(name)
                if kp and kp[2] > 0.3:
                    points.append((kp[0], kp[1]))
        hand = (objects_state or {}).get("operator_hand")
        if hand is not None and hand.bbox is not None:
            points.append(hand.bbox.centroid)
        return points

    @staticmethod
    def _interaction(bbox: List[float], hands: List[Tuple[float, float]]) -> str:
        if not hands:
            return "NONE"
        x1, y1, x2, y2 = bbox
        diag = math.hypot(x2 - x1, y2 - y1)
        best = float("inf")
        for hx, hy in hands:
            dx = max(x1 - hx, 0.0, hx - x2)
            dy = max(y1 - hy, 0.0, hy - y2)
            best = min(best, math.hypot(dx, dy))
        if best == 0.0:
            return "HOLDING"
        if best < max(25.0, 0.15 * diag):
            return "TOUCHING"
        if best < max(80.0, 0.5 * diag):
            return "NEAR"
        return "NONE"

    @staticmethod
    def summarize(detections: List[Dict[str, Any]]) -> Optional[str]:
        """Short activity text for the strongest interaction, e.g. 'HOLDING CUP'."""
        for det in detections:
            if det["interaction"] in ("HOLDING", "TOUCHING"):
                return f"{det['interaction']} {det['label'].upper()}"
        return None

    @staticmethod
    def draw(frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """Draws labelled boxes in place; interacting objects are highlighted."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            inter = det["interaction"]
            color = _HOLD_COLOR if inter == "HOLDING" else _TOUCH_COLOR if inter == "TOUCHING" else _IDLE_COLOR
            thickness = 2 if inter in ("HOLDING", "TOUCHING") else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            text = f"{det['label']} {det['conf']:.2f}" + (f" [{inter}]" if inter != "NONE" else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(frame, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

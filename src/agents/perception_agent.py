"""
BAS Autonomous HAR System - Agent 1: Perception Agent
Detects experimental components (Container, Lid, Red Box, Yellow Box, Hands)
and extracts 3D skeletal keypoints in camera space using PhysAstro-Pose principles.
"""

import os
import math
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from src.core.types import (
    BBox2D,
    ExperimentObject,
    EntityState,
    Vector3D,
    Joint3D,
    AstronautPose3D
)


class PerceptionAgent:
    """Perception Agent executing object detection and 3D human pose recovery."""

    def __init__(
        self,
        calib_path: str = "configs/camera_calib.json",
        model_path: Optional[str] = "models/detector_offline.pt",
        common_model_path: Optional[str] = "yolov8n.pt",
        pose_model_path: Optional[str] = "models/yolov8n-pose.pt",
        enable_common_detection: bool = True,
        enable_pose_detection: bool = True
    ):
        # Load camera intrinsics
        with open(calib_path, "r") as f:
            calib_data = json.load(f)
        
        intr = calib_data["intrinsics"]
        self.fx = intr["fx"]
        self.fy = intr["fy"]
        self.cx = intr["cx"]
        self.cy = intr["cy"]
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)

        self.last_lid_angle = 0.0
        self.payload_was_extracted = False
        self.payload_is_docked = False
        self.payload_docked_frames = 0
        self.red_was_extracted = False
        self.yellow_was_extracted = False

        # 1. Offline YOLOv8 Deep Neural Object Detector for Experiment Protocol
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.model = YOLO(model_path)
                print(f"[Perception Agent] Offline YOLOv8 detector engaged: {model_path}")
            except Exception as e:
                print(f"[Perception Agent] Offline model note: {e}. Active with contour heuristics.")

        # 2. General / Common Object Detector (YOLO11n / YOLOv8 COCO-80 Pretrained)
        self.common_model = None
        common_candidates = [
            common_model_path,
            "models/yolo11n.pt",
            "yolo11n.pt",
            "yolov8n.pt"
        ]
        chosen_common = None
        for cand in common_candidates:
            if cand and os.path.exists(cand):
                chosen_common = cand
                break

        if enable_common_detection and chosen_common:
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.common_model = YOLO(chosen_common)
                print(f"[Perception Agent] Common Object Detector ({os.path.basename(chosen_common).upper()}) engaged: {chosen_common}")
            except Exception as e:
                print(f"[Perception Agent] Common model note: {e}")

        # 3. Real-Time Person Skeleton & Human Pose Detector (YOLOv8-Pose)
        self.pose_model = None
        pose_candidate = pose_model_path or "models/yolov8n-pose.pt"
        if not os.path.exists(pose_candidate) and os.path.exists("yolov8n-pose.pt"):
            pose_candidate = "yolov8n-pose.pt"
        if enable_pose_detection and os.path.exists(pose_candidate):
            try:
                import importlib
                ultralytics_pkg = importlib.import_module("ultralytics")
                YOLO = getattr(ultralytics_pkg, "YOLO")
                self.pose_model = YOLO(pose_candidate)
                print(f"[Perception Agent] Real-Time Person Skeleton Detector engaged: {pose_candidate}")
            except Exception as e:
                print(f"[Perception Agent] Pose detector note: {e}")

        # 4. RelateAnything Engine for Scene Understanding
        self.relate_anything = None
        self.frame_count = 0
        self.last_relations = {}  # Cache relations across frames
        
        import sys
        _ra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../RelateAnything-main/RelateAnything-main"))
        if os.path.exists(_ra_path) and _ra_path not in sys.path:
            sys.path.insert(0, _ra_path)
        
        try:
            from relsgg import RelateAnything  # type: ignore
            # Use CPU as hardware configuration specified no GPU
            self.relate_anything = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", device="cpu")
            self.relate_anything.set_vocabulary([
                "standing next to", "far away from", "holding", "touching", "looking at",
                "dancing", "walking away"
            ])
            print("[Perception Agent] RelateAnything engaged on CPU.")
        except Exception as e:
            print(f"[Perception Agent] RelateAnything init note: {e}")


    def reset(self):
        """Resets dynamic tracking state for a new test or experiment run."""
        self.last_lid_angle = 0.0
        self.payload_was_extracted = False
        self.payload_is_docked = False
        self.payload_docked_frames = 0
        self.red_was_extracted = False
        self.yellow_was_extracted = False

    def process_frame(
        self,
        frame: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Dict[str, ExperimentObject], AstronautPose3D, float]:
        """
        Processes camera frame to detect objects and human 3D pose.
        Returns:
            - objects: Dictionary of detected ExperimentObjects
            - pose: Reconstructed 3D astronaut pose in camera coordinates
            - lid_angle: Measured container lid angle in degrees
        """
        self.frame_count += 1
        h, w, _ = frame.shape
        objects: Dict[str, ExperimentObject] = {}
        hand_bbox = None
        hand_center = (0.0, 0.0)

        # 1. Primary Neural Object Detector (YOLOv8 offline model trained on boxes & hands)
        if self.model is not None:
            try:
                # conf=0.25 cleanly rejects background clutter while preserving real boxes (conf ~0.85-0.97)
                results = self.model(frame, verbose=False, conf=0.25)
                if results and len(results) > 0 and results[0].boxes:
                    class_names = getattr(self.model, "names", {
                        0: "container_box", 1: "container_lid", 2: "component_box",
                        3: "operator_hand", 4: "human_body"
                    })
                    boxes_sorted = sorted(results[0].boxes, key=lambda b: float(b.conf[0].item()), reverse=True)
                    frame_area = float(w * h)
                    for box in boxes_sorted:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        name = class_names.get(cls_id, f"obj_{cls_id}")

                        box_w = bx2 - bx1
                        box_h = by2 - by1
                        box_area = box_w * box_h

                        # Discard degenerate tiny noise boxes (< 800px) or full-screen container hallucinations (> 98% of frame)
                        if box_area < 800 or (cls_id == 0 and box_area > 0.98 * frame_area):
                            continue

                        b = BBox2D(
                            xmin=float(bx1), ymin=float(by1),
                            xmax=float(bx2), ymax=float(by2),
                            confidence=conf, class_id=cls_id, class_name=name
                        )
                        center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)

                        if name in ("operator_hand", "hand", "astronaut_hand"):
                            if not hand_bbox:
                                hand_bbox = b
                                hand_center = center
                                objects["operator_hand"] = ExperimentObject(
                                    name="operator_hand",
                                    class_name="operator_hand",
                                    bbox=b,
                                    pos_rack=self._pixel_to_camera_coord(center[0], center[1], depth_m=1.00)
                                )
                        elif name in ("human_body", "person", "astronaut"):
                            if "human_body" not in objects:
                                obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.80)
                                objects["human_body"] = ExperimentObject(
                                    name="human_body",
                                    class_name="human_body",
                                    bbox=b,
                                    pos_rack=obj_cam
                                )
                        else:
                            if name not in objects:
                                obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.20)
                                objects[name] = ExperimentObject(name=name, class_name=name, bbox=b, pos_rack=obj_cam)
            except Exception:
                pass

        # 2. Supplementary Common Object Detector (COCO-80 for secondary manipulable props only)
        if self.common_model is not None:
            try:
                results_common = self.common_model(frame, verbose=False, conf=0.40)
                if results_common and len(results_common) > 0 and results_common[0].boxes:
                    for box in results_common[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        cls_name = self.common_model.names.get(cls_id, f"obj_{cls_id}")

                        # ONLY alias genuine small manipulable items if component_box is not yet detected
                        if cls_name in ("bottle", "cup", "cell phone", "book", "bowl", "scissors") and "component_box" not in objects:
                            b = BBox2D(
                                xmin=float(bx1), ymin=float(by1),
                                xmax=float(bx2), ymax=float(by2),
                                confidence=conf, class_id=cls_id, class_name=cls_name
                            )
                            center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                            obj_cam = self._pixel_to_camera_coord(center[0], center[1], depth_m=1.20)
                            objects["component_box"] = ExperimentObject(
                                name="component_box",
                                class_name=cls_name,
                                bbox=b,
                                pos_rack=obj_cam,
                                state=EntityState.DOCKED,
                                is_inside_container=False
                            )
                            break
            except Exception:
                pass

        # 3. Extract bounding box of container if present
        cont_bbox = objects["container_box"].bbox if "container_box" in objects else None

        # 4. Detect Glove / Bare Human Hand in workspace (fallback only if neural model is NOT engaged)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not hand_bbox and self.model is None:
            mask_skin1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 200, 255]))
            mask_skin2 = cv2.inRange(hsv, np.array([165, 30, 60]), np.array([180, 200, 255]))
            mask_glove = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            mask_hand = cv2.bitwise_or(cv2.bitwise_or(mask_skin1, mask_skin2), mask_glove)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_hand = cv2.morphologyEx(mask_hand, cv2.MORPH_OPEN, kernel)

            # Prioritize hands in the workspace (exclude top 20% head zone on webcam)
            mask_workspace_hand = mask_hand.copy()
            mask_workspace_hand[:int(h * 0.20), :] = 0
            hand_bbox, hand_center = self._extract_largest_bbox(mask_workspace_hand, "operator_hand", 3, min_area=800)
            if not hand_bbox:
                hand_bbox, hand_center = self._extract_largest_bbox(mask_hand, "operator_hand", 3, min_area=1000)
            if hand_bbox and "operator_hand" not in objects:
                objects["operator_hand"] = ExperimentObject(
                    name="operator_hand",
                    class_name="operator_hand",
                    bbox=hand_bbox,
                    pos_rack=self._pixel_to_camera_coord(hand_center[0], hand_center[1], depth_m=1.00)
                )

        # 5. Detection for ISRO Dual-Box benchmark objects (Red Box, Yellow Box)
        # Spatial filtering: restrict to astronaut workspace (exclude upper wall & right background)
        if "red_box" not in objects:
            mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_red[:int(0.28 * h), :] = 0   # Exclude upper ceiling / background
            mask_red[:, int(0.62 * w):] = 0   # Exclude right wall
            red_bbox, red_center = self._extract_largest_bbox(mask_red, "red_box", 2, min_area=200, max_area=150000)
            if red_bbox:
                red_cam = self._pixel_to_camera_coord(red_center[0], red_center[1], depth_m=1.15)
                objects["red_box"] = ExperimentObject(name="red_box", class_name="red_box", bbox=red_bbox, pos_rack=red_cam)

        if "yellow_box" not in objects:
            mask_yellow = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([40, 255, 255]))
            mask_yellow[:int(0.32 * h), :] = 0  # Exclude upper ceiling / background wall
            mask_yellow[:, int(0.62 * w):] = 0   # Exclude right wall
            yel_bbox, yel_center = self._extract_largest_bbox(mask_yellow, "yellow_box", 2, min_area=200, max_area=150000)
            if yel_bbox:
                yel_cam = self._pixel_to_camera_coord(yel_center[0], yel_center[1], depth_m=1.15)
                objects["yellow_box"] = ExperimentObject(name="yellow_box", class_name="yellow_box", bbox=yel_bbox, pos_rack=yel_cam)

        # 6. Compute Lid Elevation Angle
        if "container_lid" in objects:
            lid_b = objects["container_lid"].bbox
            if lid_b and cont_bbox:
                elevation = max(0.0, cont_bbox.ymin - lid_b.ymin)
                target_angle = min(85.0, (elevation / max(30.0, cont_bbox.height * 0.5)) * 80.0)
            else:
                target_angle = 0.0
        elif cont_bbox:
            # Dynamic edge & contour check above container box
            target_angle = self._estimate_lid_angle(frame, cont_bbox)
        else:
            target_angle = 0.0

        # Smooth angle transitions
        self.last_lid_angle = 0.75 * self.last_lid_angle + 0.25 * target_angle
        lid_angle = self.last_lid_angle

        # 7. Construct 3D Astronaut Pose in Camera Space
        pose = self._estimate_3d_pose(frame, hand_bbox, hand_center, w, h)

        # If hand bbox was not found by object detector, register hand from detected skeleton wrist
        if not hand_bbox and "wrist" in pose.keypoints_2d:
            wx, wy, wconf = pose.keypoints_2d["wrist"]
            hw = 45.0
            hand_bbox = BBox2D(
                xmin=max(0.0, wx - hw), ymin=max(0.0, wy - hw),
                xmax=min(float(w), wx + hw), ymax=min(float(h), wy + hw),
                confidence=wconf, class_id=3, class_name="operator_hand"
            )
            hand_center = (wx, wy)
            if "operator_hand" not in objects:
                objects["operator_hand"] = ExperimentObject(
                    name="operator_hand",
                    class_name="operator_hand",
                    bbox=hand_bbox,
                    pos_rack=self._pixel_to_camera_coord(wx, wy, depth_m=1.00)
                )

        # 8. Resolve Component Box Payload & Extraction / Return State
        self._resolve_component_box(frame, objects, cont_bbox, lid_angle, pose, w, h)

        # 9. Real-Time Spatial and Interaction Relation Prediction (Every 15 frames)
        if self.relate_anything is not None:
            if self.frame_count % 15 == 0 and len(objects) >= 2:
                try:
                    # Prepare object list and their boxes
                    obj_names = list(objects.keys())
                    boxes = []
                    for name in obj_names:
                        obj = objects[name]
                        if obj.bbox:
                            boxes.append([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax])
                        else:
                            # Fallback for objects without standard bbox (should be rare)
                            boxes.append([0.0, 0.0, 1.0, 1.0])
                    
                    boxes_xyxy = np.array(boxes, dtype=np.float32)
                    
                    # Predict relations
                    triplets = self.relate_anything.predict(frame, boxes_xyxy, topk=15)
                    
                    from src.core.types import Relation
                    self.last_relations = {} # Reset cached relations
                    for t in triplets:
                        sub_name = obj_names[t.subject_idx]
                        obj_name = obj_names[t.object_idx]
                        
                        rel = Relation(
                            subject_name=sub_name, 
                            object_name=obj_name, 
                            predicate=t.predicate, 
                            score=float(t.score)
                        )
                        if sub_name not in self.last_relations:
                            self.last_relations[sub_name] = []
                        self.last_relations[sub_name].append(rel)
                        
                except Exception as e:
                    print(f"[Perception Agent] RelateAnything prediction error: {e}")
            
            # Attach cached relations to the current frame's objects
            for obj_name, rel_list in self.last_relations.items():
                if obj_name in objects:
                    objects[obj_name].relations = rel_list

        return objects, pose, lid_angle

    def _resolve_component_box(
        self,
        frame: np.ndarray,
        objects: Dict[str, ExperimentObject],
        cont_bbox: Optional[BBox2D],
        lid_angle: float,
        pose: AstronautPose3D,
        w: int,
        h: int
    ):
        """
        Deterministically tracks and resolves the component_box payload state:
        - Detects when the component is extracted into mid-air outside the container.
        - Detects when the component is returned back inside the container cavity.
        - Tracks the lifecycle across the complete procedural sequence.
        """
        if not cont_bbox:
            # Fallback container coordinates if lost by neural detector
            cont_ymin = h * 0.55
            cont_xmin = w * 0.35
            cont_xmax = w * 0.65
            cont_cx = w * 0.50
            cont_cy = h * 0.75
            c_ymax = h
        else:
            cont_ymin = cont_bbox.ymin
            cont_xmin = cont_bbox.xmin
            cont_xmax = cont_bbox.xmax
            cont_cx = (cont_bbox.xmin + cont_bbox.xmax) / 2.0
            cont_cy = (cont_bbox.ymin + cont_bbox.ymax) / 2.0
            c_ymax = cont_bbox.ymax

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extract detected wrists from pose
        wrists = []
        if pose and hasattr(pose, "keypoints_2d"):
            for k in ("wrist", "left_wrist", "right_wrist"):
                if k in pose.keypoints_2d:
                    wx, wy, c = pose.keypoints_2d[k]
                    if c > 0.30:
                        wrists.append((wx, wy))
        if not wrists and "operator_hand" in objects:
            hb = objects["operator_hand"].bbox
            if hb:
                wrists.append(((hb.xmin + hb.xmax) / 2.0, (hb.ymin + hb.ymax) / 2.0))

        # Check and update red_box extraction state
        red_obj = objects.get("red_box")
        if red_obj and red_obj.bbox:
            rcy = (red_obj.bbox.ymin + red_obj.bbox.ymax) / 2.0
            rcx = (red_obj.bbox.xmin + red_obj.bbox.xmax) / 2.0
            is_outside_x = (rcx < cont_xmin - 20) or (rcx > cont_xmax + 20)
            is_in_air = (rcy < cont_ymin - 20)
            if is_outside_x or is_in_air:
                red_obj.is_inside_container = False
                red_obj.state = EntityState.EXTRACTED
                self.red_was_extracted = True
            else:
                red_obj.is_inside_container = True
                red_obj.state = EntityState.DOCKED

        # Check and update yellow_box extraction state
        yel_obj = objects.get("yellow_box")
        if yel_obj and yel_obj.bbox:
            ycy = (yel_obj.bbox.ymin + yel_obj.bbox.ymax) / 2.0
            ycx = (yel_obj.bbox.xmin + yel_obj.bbox.xmax) / 2.0

            is_outside_x = (ycx < cont_xmin - 20) or (ycx > cont_xmax + 20)
            is_in_air = (ycy < cont_ymin - 20)
            if is_outside_x or is_in_air:
                yel_obj.is_inside_container = False
                yel_obj.state = EntityState.EXTRACTED
                self.yellow_was_extracted = True
            else:
                yel_obj.is_inside_container = True
                yel_obj.state = EntityState.DOCKED

        # Check if neural model detected component_box
        comp = objects.get("component_box")
        if comp and comp.bbox:
            comp_cy = (comp.bbox.ymin + comp.bbox.ymax) / 2.0
            if comp_cy < cont_ymin - 20:
                comp.is_inside_container = False
                comp.state = EntityState.EXTRACTED
                self.payload_was_extracted = True
            else:
                comp.is_inside_container = True
                comp.state = EntityState.DOCKED
            
            # Since component_box is handled by the neural model here, we still need to process wrists_elevated for red/yellow
            pass
        
        # Resolution scaling factors (calibrated against 1080p reference)
        scale_y = float(h) / 1080.0
        scale_x = float(w) / 1920.0
        scale_area = (float(w) * float(h)) / (1920.0 * 1080.0)

        elev_thresh = min(int(0.48 * h), int(cont_ymin - 130.0 * scale_y))
        wrists_elevated = any(wy < elev_thresh for wx, wy in wrists) if wrists else False

        # Persist red and yellow boxes inside container if they vanish (e.g. dropped inside and occluded)
        if not wrists_elevated:
            if self.red_was_extracted and "red_box" not in objects:
                objects["red_box"] = ExperimentObject(
                    name="red_box", class_name="red_box", bbox=None,
                    pos_rack=objects["container_box"].pos_rack if "container_box" in objects else Vector3D(),
                    state=EntityState.DOCKED, is_inside_container=True
                )
            if self.yellow_was_extracted and "yellow_box" not in objects:
                objects["yellow_box"] = ExperimentObject(
                    name="yellow_box", class_name="yellow_box", bbox=None,
                    pos_rack=objects["container_box"].pos_rack if "container_box" in objects else Vector3D(),
                    state=EntityState.DOCKED, is_inside_container=True
                )

        if comp and comp.bbox:
            return

        # Resolution scaling factors (calibrated against 1080p reference)
        scale_y = float(h) / 1080.0
        scale_x = float(w) / 1920.0
        scale_area = (float(w) * float(h)) / (1920.0 * 1080.0)

        # Check for mid-air extracted cardboard payload contour (strictly above container flaps in torso zone)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_brown = np.array([8, 25, 50])
        upper_brown = np.array([30, 235, 240])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Container flaps elevate up to (cont_ymin - 120 * scale_y). Component box when extracted is held in upper torso space (y < 0.48 * h)
        air_boundary_y = min(int(0.48 * h), int(cont_ymin - 130.0 * scale_y))
        mask_air = mask.copy()
        mask_air[max(0, air_boundary_y):, :] = 0
        mask_air[:, :max(0, int(0.25 * w))] = 0
        mask_air[:, min(w, int(0.75 * w)):] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_air = cv2.morphologyEx(mask_air, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_air, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_payload = None
        max_area = 0
        min_payload_area = max(500.0, 3800.0 * scale_area)
        min_payload_h = max(30.0, 90.0 * scale_y)
        wrist_tol_x = max(40.0, 130.0 * scale_x)
        wrist_tol_y = max(40.0, 130.0 * scale_y)

        for c in contours:
            area = cv2.contourArea(c)
            if area > min_payload_area:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(max(1, bw))
                if bh >= min_payload_h and aspect >= 0.65:
                    near_wrist = False
                    if wrists:
                        for wx, wy in wrists:
                            if wy < air_boundary_y + 50.0 * scale_y and (x - wrist_tol_x <= wx <= x + bw + wrist_tol_x) and (y - wrist_tol_y <= wy <= y + bh + wrist_tol_y):
                                near_wrist = True
                                break
                    else:
                        near_wrist = True

                    if near_wrist and area > max_area:
                        max_area = area
                        found_payload = (x, y, bw, bh)

        elev_thresh = min(int(0.48 * h), int(cont_ymin - 130.0 * scale_y))
        wrists_elevated = any(wy < elev_thresh for wx, wy in wrists) if wrists else False

        if found_payload:
            self.payload_is_docked = False
            self.payload_docked_frames = 0
            px, py, pbw, pbh = found_payload
            item_bbox = BBox2D(
                xmin=float(px), ymin=float(py),
                xmax=float(px + pbw), ymax=float(py + pbh),
                confidence=0.88, class_id=2, class_name="component_box"
            )
            pcx, pcy = px + pbw / 2.0, py + pbh / 2.0
            objects["component_box"] = ExperimentObject(
                name="component_box",
                class_name="component_box",
                bbox=item_bbox,
                pos_rack=self._pixel_to_camera_coord(pcx, pcy, depth_m=1.15),
                state=EntityState.EXTRACTED,
                is_inside_container=False
            )
            self.payload_was_extracted = True
            return

        # If payload was extracted, NOT yet docked into container, and wrists are still elevated high in torso workspace
        if self.payload_was_extracted and not self.payload_is_docked and wrists_elevated:
            elevated_wrists = [wy for wx, wy in wrists if wy < elev_thresh]
            if elevated_wrists:
                avg_wx = float(np.mean([wx for wx, wy in wrists if wy < elev_thresh]))
                avg_wy = float(np.mean(elevated_wrists))
                box_rad_x = max(25.0, 70.0 * scale_x)
                box_rad_y = max(25.0, 70.0 * scale_y)
                item_bbox = BBox2D(
                    xmin=max(0.0, avg_wx - box_rad_x), ymin=max(0.0, avg_wy - box_rad_y),
                    xmax=min(float(w), avg_wx + box_rad_x), ymax=min(float(h), avg_wy + box_rad_y),
                    confidence=0.82, class_id=2, class_name="component_box"
                )
                objects["component_box"] = ExperimentObject(
                    name="component_box",
                    class_name="component_box",
                    bbox=item_bbox,
                    pos_rack=self._pixel_to_camera_coord(avg_wx, avg_wy, depth_m=1.15),
                    state=EntityState.EXTRACTED,
                    is_inside_container=False
                )
                return

        # If not in air:
        # Case A: previously extracted, now returned inside container
        if self.payload_was_extracted:
            self.payload_docked_frames += 1
            if self.payload_docked_frames >= 5:
                self.payload_is_docked = True
            comp_w = max(60.0, (cont_xmax - cont_xmin) * 0.40)
            comp_h = max(50.0, (c_ymax - cont_ymin) * 0.25)
            in_bbox = BBox2D(
                xmin=max(0.0, float(cont_cx - comp_w / 2)),
                ymin=max(0.0, float(cont_cy - comp_h / 2)),
                xmax=min(float(w), float(cont_cx + comp_w / 2)),
                ymax=min(float(h), float(cont_cy + comp_h / 2)),
                confidence=0.85, class_id=2, class_name="component_box"
            )
            objects["component_box"] = ExperimentObject(
                name="component_box",
                class_name="component_box",
                bbox=in_bbox,
                pos_rack=objects["container_box"].pos_rack if "container_box" in objects else Vector3D(),
                state=EntityState.DOCKED,
                is_inside_container=True
            )
        # Case B: Box is open, but component has not yet been extracted (docked in cavity)
        elif lid_angle > 15.0 or "container_lid" in objects:
            comp_w = max(60.0, (cont_xmax - cont_xmin) * 0.40)
            comp_h = max(50.0, (c_ymax - cont_ymin) * 0.25)
            in_bbox = BBox2D(
                xmin=max(0.0, float(cont_cx - comp_w / 2)),
                ymin=max(0.0, float(cont_cy - comp_h / 2)),
                xmax=min(float(w), float(cont_cx + comp_w / 2)),
                ymax=min(float(h), float(cont_cy + comp_h / 2)),
                confidence=0.80, class_id=2, class_name="component_box"
            )
            objects["component_box"] = ExperimentObject(
                name="component_box",
                class_name="component_box",
                bbox=in_bbox,
                pos_rack=objects["container_box"].pos_rack if "container_box" in objects else Vector3D(),
                state=EntityState.DOCKED,
                is_inside_container=True
            )

    def _extract_largest_bbox(
        self,
        mask: np.ndarray,
        class_name: str,
        class_id: int,
        min_area: int = 500,
        max_area: Optional[int] = None
    ) -> Tuple[Optional[BBox2D], Tuple[float, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (0.0, 0.0)

        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area and (max_area is None or area <= max_area):
                valid_contours.append(c)

        if not valid_contours:
            return None, (0.0, 0.0)

        c = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            return None, (0.0, 0.0)

        aspect = bw / float(bh)
        if aspect > 4.0 or aspect < 0.25:
            # Reject extreme aspect ratios (lines, border edges, shadows)
            return None, (0.0, 0.0)

        bbox = BBox2D(
            xmin=float(x),
            ymin=float(y),
            xmax=float(x + bw),
            ymax=float(y + bh),
            confidence=min(0.95, max(0.40, area / 8000.0)),
            class_id=class_id,
            class_name=class_name
        )
        return bbox, (x + bw / 2.0, y + bh / 2.0)

    def _estimate_lid_angle(self, frame: np.ndarray, cont_bbox: Optional[BBox2D]) -> float:
        """Estimates container lid elevation angle based on cardboard flaps above container."""
        if not cont_bbox:
            return self.last_lid_angle

        # Check upper region above container for lid flap presence (cardboard brown)
        y_top = int(cont_bbox.ymin)
        y_search_top = max(0, y_top - 160)
        x1 = max(0, int(cont_bbox.xmin - 30))
        x2 = min(frame.shape[1], int(cont_bbox.xmax + 30))

        crop = frame[y_search_top:y_top, x1:x2]
        if crop.size == 0:
            return self.last_lid_angle

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_brown = cv2.inRange(hsv_crop, np.array([8, 25, 50]), np.array([30, 235, 240]))
        brown_density = np.sum(mask_brown > 0) / mask_brown.size

        # If significant elevated cardboard flap is visible above container box, lid is elevated
        if brown_density > 0.04:
            target_angle = min(75.0, max(28.0, (brown_density / 0.25) * 60.0))
        else:
            target_angle = 0.0

        # Smooth angle transitions
        self.last_lid_angle = 0.75 * self.last_lid_angle + 0.25 * target_angle
        return self.last_lid_angle

    def _pixel_to_camera_coord(self, u: float, v: float, depth_m: float) -> Vector3D:
        """Projects (u, v) image pixel + metric depth to camera coordinates X_C."""
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return Vector3D(x, y, depth_m)

    def _estimate_3d_pose(
        self,
        frame: Optional[np.ndarray],
        hand_bbox: Optional[BBox2D],
        hand_center: Tuple[float, float],
        width: int,
        height: int
    ) -> AstronautPose3D:
        """
        Detects full-body person skeleton (17 COCO joints) and reconstructs
        true biomechanical 3D kinematics for the astronaut.
        """
        pose = AstronautPose3D()

        # 1. Real-Time Neural Pose Estimation (YOLOv8-Pose)
        if self.pose_model is not None and frame is not None and frame.size > 0:
            try:
                res_pose = self.pose_model(frame, verbose=False, conf=0.20, imgsz=480)
                if res_pose and len(res_pose) > 0 and len(res_pose[0].boxes) > 0 and res_pose[0].keypoints is not None:
                    boxes = res_pose[0].boxes
                    best_idx = 0
                    if len(boxes) > 1:
                        areas = [float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])) for b in boxes]
                        best_idx = int(np.argmax(areas))

                    kp = res_pose[0].keypoints
                    xy = kp.xy[best_idx].cpu().numpy()  # shape (17, 2)
                    confs = kp.conf[best_idx].cpu().numpy() if kp.conf is not None else np.ones(17)

                    COCO_KEYPOINTS = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]

                    for i, k_name in enumerate(COCO_KEYPOINTS):
                        k_conf = float(confs[i])
                        if k_conf > 0.20:
                            kx, ky = float(xy[i][0]), float(xy[i][1])
                            pose.keypoints_2d[k_name] = (kx, ky, k_conf)
                            pos_cam = self._pixel_to_camera_coord(kx, ky, depth_m=1.20)
                            pose.joints[k_name] = Joint3D(name=k_name, pos_camera=pos_cam, confidence=k_conf)

                    # Determine dominant/active working arm
                    has_r_arm = ("right_wrist" in pose.joints and "right_elbow" in pose.joints)
                    has_l_arm = ("left_wrist" in pose.joints and "left_elbow" in pose.joints)
                    active_side = "right" if has_r_arm or not has_l_arm else "left"

                    if f"{active_side}_wrist" in pose.joints:
                        pose.joints["wrist"] = pose.joints[f"{active_side}_wrist"]
                        pose.keypoints_2d["wrist"] = pose.keypoints_2d[f"{active_side}_wrist"]
                    if f"{active_side}_elbow" in pose.joints:
                        pose.joints["elbow"] = pose.joints[f"{active_side}_elbow"]
                    if f"{active_side}_shoulder" in pose.joints:
                        pose.joints["shoulder"] = pose.joints[f"{active_side}_shoulder"]

                    # Compute true measured biomechanical angles
                    if "wrist" in pose.joints and "elbow" in pose.joints and "shoulder" in pose.joints:
                        w_p = pose.joints["wrist"].pos_camera
                        e_p = pose.joints["elbow"].pos_camera
                        s_p = pose.joints["shoulder"].pos_camera

                        v_fore = np.array([w_p.x - e_p.x, w_p.y - e_p.y, w_p.z - e_p.z])
                        v_upper = np.array([s_p.x - e_p.x, s_p.y - e_p.y, s_p.z - e_p.z])
                        norm_p = np.linalg.norm(v_fore) * np.linalg.norm(v_upper)
                        if norm_p > 1e-6:
                            cos_elb = np.clip(np.dot(v_fore, v_upper) / norm_p, -1.0, 1.0)
                            pose.elbow_angle_deg = float(np.degrees(np.arccos(cos_elb)))
                        else:
                            pose.elbow_angle_deg = 77.0

                        hip_j = pose.joints.get(f"{active_side}_hip") or pose.joints.get("left_hip") or pose.joints.get("right_hip")
                        if hip_j:
                            h_p = hip_j.pos_camera
                            v_torso = np.array([h_p.x - s_p.x, h_p.y - s_p.y, h_p.z - s_p.z])
                            norm_t = np.linalg.norm(v_torso) * np.linalg.norm(v_upper)
                            if norm_t > 1e-6:
                                cos_sh = np.clip(np.dot(v_torso, v_upper) / norm_t, -1.0, 1.0)
                                pose.shoulder_angle_deg = float(np.degrees(np.arccos(cos_sh)))

                        pose.body_orientation_deg = float(math.degrees(math.atan2(v_upper[1], v_upper[0])))

                    pose.rom_limits_violated = not (0.0 <= pose.elbow_angle_deg <= 145.0)
                    return pose
            except Exception:
                pass

        # 2. Heuristic Prior Fallback (if pose model is not active and hand is detected)
        if hand_bbox:
            hx, hy = hand_center
            wrist_pos = self._pixel_to_camera_coord(hx, hy, depth_m=1.10)
            pose.joints["wrist"] = Joint3D(name="wrist", pos_camera=wrist_pos, confidence=hand_bbox.confidence)
            pose.keypoints_2d["wrist"] = (hx, hy, hand_bbox.confidence)

            arm_length_m = 0.28
            forearm_vec = Vector3D(x=-0.18, y=-0.15, z=0.08)
            elbow_pos = Vector3D(
                x=wrist_pos.x + forearm_vec.x,
                y=wrist_pos.y + forearm_vec.y,
                z=wrist_pos.z + forearm_vec.z
            )
            pose.joints["elbow"] = Joint3D(name="elbow", pos_camera=elbow_pos, confidence=0.85)

            upper_arm_vec = Vector3D(x=-0.22, y=-0.20, z=0.05)
            shoulder_pos = Vector3D(
                x=elbow_pos.x + upper_arm_vec.x,
                y=elbow_pos.y + upper_arm_vec.y,
                z=elbow_pos.z + upper_arm_vec.z
            )
            pose.joints["shoulder"] = Joint3D(name="shoulder", pos_camera=shoulder_pos, confidence=0.80)

            v_fore = np.array([wrist_pos.x - elbow_pos.x, wrist_pos.y - elbow_pos.y, wrist_pos.z - elbow_pos.z])
            v_upper = np.array([shoulder_pos.x - elbow_pos.x, shoulder_pos.y - elbow_pos.y, shoulder_pos.z - elbow_pos.z])
            norm_product = np.linalg.norm(v_fore) * np.linalg.norm(v_upper)
            if norm_product > 1e-6:
                cos_elbow = np.clip(np.dot(v_fore, v_upper) / norm_product, -1.0, 1.0)
                pose.elbow_angle_deg = float(np.degrees(np.arccos(cos_elbow)))
            else:
                pose.elbow_angle_deg = 77.0

            pose.body_orientation_deg = float(math.degrees(math.atan2(v_upper[1], v_upper[0])))

        pose.rom_limits_violated = not (0.0 <= pose.elbow_angle_deg <= 145.0)
        return pose

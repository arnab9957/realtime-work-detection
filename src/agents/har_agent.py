"""
BAS Autonomous HAR System - Agent 4: HAR Agent
Evaluates Hand-Object Interaction (HOI) spatial metrics, AdaSpot saliency gating,
and temporal activity recognition primitives (Approach, Contact, Grasp, Extract, Release).
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    HOIInteraction,
    HOIAction,
    EntityState,
    Vector3D,
    SpatialMetrics
)


class HARAgent:
    """Human Activity Recognition (HAR) & Hand-Object Interaction (HOI) Agent."""

    EXCLUDED_TARGETS = {"operator_hand", "hand", "person", "astronaut", "human_body"}

    def __init__(self, contact_threshold_m: float = 0.12):
        self.contact_threshold_m = contact_threshold_m
        self.approach_threshold_m = 0.28
        
        # State tracking per object
        self.contact_frame_counters: Dict[str, int] = {
            "container_box": 0,
            "container_lid": 0,
            "component_box": 0
        }
        self.previously_extracted: Dict[str, bool] = {}
        self.extraction_threshold_y = 0.20 # Metric meters offset relative to container
        self.previous_wrist_pos: Optional[Vector3D] = None
        self.last_lid_angle: float = 0.0
        self.pose_history: List[Dict[str, Any]] = []
        self.dance_frames: int = 0
        self.non_proc_frames: int = 0
        self.active_non_proc_act: Optional[str] = None

    def reset(self):
        """Resets all HOI contact counters for a new test cycle."""
        for k in self.contact_frame_counters:
            self.contact_frame_counters[k] = 0
        self.previously_extracted.clear()
        self.previous_wrist_pos = None
        self.last_lid_angle = 0.0
        self.pose_history.clear()
        self.dance_frames = 0
        self.non_proc_frames = 0
        self.active_non_proc_act = None

    def evaluate_interactions(
        self,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        spatial_metrics: SpatialMetrics
    ) -> Tuple[List[HOIInteraction], Dict[str, ExperimentObject], str]:
        """
        Evaluates 3D spatial kinematics and interactions between astronaut hand and experimental items.
        Returns:
            - active_hoi: list of HOIInteraction items
            - updated_objects: objects with updated EntityStates
            - primary_activity: active action string (e.g. 'OPEN LID', 'EXTRACT COMPONENT', 'IDLE')
        """
        active_hoi: List[HOIInteraction] = []
        primary_activity = "IDLE"

        wrist = pose.joints.get("wrist")
        if not wrist:
            self.last_lid_angle = lid_angle
            return active_hoi, objects, primary_activity

        wrist_pos = wrist.pos_rack
        cont = objects.get("container_box")
        cont_pos = cont.pos_rack if cont else Vector3D()

        # Identify manipulable target objects, strictly excluding human hands or person boxes
        target_names = [k for k in objects.keys() if k not in self.EXCLUDED_TARGETS and k != "container_box"]

        # Track lid state transitions
        delta_lid = lid_angle - self.last_lid_angle
        self.last_lid_angle = lid_angle

        # 1. Evaluate interactions with manipulable objects (component_box, red_box, etc.)
        for obj_name in target_names:
            obj = objects.get(obj_name)
            if not obj:
                continue

            dist = spatial_metrics.distance_to_components_m.get(obj_name, 999.0)
            if obj_name not in self.contact_frame_counters:
                self.contact_frame_counters[obj_name] = 0

            # Determine Action Primitive
            if dist <= self.contact_threshold_m:
                self.contact_frame_counters[obj_name] += 1
                if self.contact_frame_counters[obj_name] >= 3:
                    action = HOIAction.GRASP
                else:
                    action = HOIAction.CONTACT
            elif dist <= self.approach_threshold_m:
                self.contact_frame_counters[obj_name] = max(0, self.contact_frame_counters[obj_name] - 1)
                action = HOIAction.APPROACH
            else:
                self.contact_frame_counters[obj_name] = 0
                action = HOIAction.IDLE

            # Check Extraction condition for manipulable items
            if obj_name not in ("container_lid", "container_box"):
                is_outside = False
                if cont and cont.bbox and obj.bbox:
                    obj_cx = (obj.bbox.xmin + obj.bbox.xmax) / 2.0
                    obj_cy = (obj.bbox.ymin + obj.bbox.ymax) / 2.0
                    # Lifted above or moved outside lateral container bounds
                    if (obj_cy < cont.bbox.ymin - 30 or 
                        obj_cx < cont.bbox.xmin - 30 or 
                        obj_cx > cont.bbox.xmax + 30):
                        is_outside = True
                    else:
                        is_outside = False
                else:
                    delta_y = obj.pos_rack.y - cont_pos.y
                    delta_x = abs(obj.pos_rack.x - cont_pos.x)
                    if delta_x > 0.22 or delta_y < -0.25:
                        is_outside = True
                    else:
                        is_outside = False

                if is_outside:
                    obj.is_inside_container = False
                    self.previously_extracted[obj_name] = True
                    if action in (HOIAction.GRASP, HOIAction.CONTACT):
                        action = HOIAction.EXTRACT
                        obj.state = EntityState.EXTRACTED
                        primary_activity = f"PICKING {obj_name.replace('_', ' ').upper()}"
                    else:
                        obj.state = EntityState.RELEASED
                        primary_activity = f"HOLDING {obj_name.replace('_', ' ').upper()}"
                else:
                    obj.is_inside_container = True
                    was_outside = self.previously_extracted.get(obj_name, False)
                    if was_outside:
                        obj.state = EntityState.DOCKED
                        primary_activity = f"RETURNING {obj_name.replace('_', ' ').upper()} INTO BOX"
                    elif action in (HOIAction.GRASP, HOIAction.CONTACT):
                        obj.state = EntityState.GRASPED
                        primary_activity = f"GRASPING {obj_name.replace('_', ' ').upper()}"
                    elif action == HOIAction.APPROACH:
                        obj.state = EntityState.APPROACHED
                        primary_activity = f"APPROACHING {obj_name.replace('_', ' ').upper()}"
                    else:
                        obj.state = EntityState.DOCKED

            if action != HOIAction.IDLE:
                active_hoi.append(HOIInteraction(
                    object_name=obj_name,
                    hand_name="right_hand",
                    distance_m=dist,
                    action=action,
                    duration_frames=self.contact_frame_counters.get(obj_name, 0)
                ))

        # 2. If no direct component interaction, evaluate container & lid interactions
        if primary_activity == "IDLE":
            # Check lid manipulation
            if lid_angle >= 15.0 and abs(delta_lid) > 0.6:
                primary_activity = "OPENING CONTAINER" if delta_lid > 0 else "CLOSING CONTAINER"
            elif cont:
                cont_dist = spatial_metrics.distance_to_container_m
                wrist_in_container = spatial_metrics.wrist_in_container_2d

                if wrist_in_container or cont_dist <= self.contact_threshold_m:
                    primary_activity = "REACHING INTO BOX" if lid_angle >= 15.0 else "CONTACTING CONTAINER"
                    active_hoi.append(HOIInteraction(
                        object_name="container_box",
                        hand_name="right_hand",
                        distance_m=cont_dist,
                        action=HOIAction.CONTACT,
                        duration_frames=1
                    ))
                elif cont_dist <= self.approach_threshold_m:
                    primary_activity = "APPROACH CONTAINER"
                    active_hoi.append(HOIInteraction(
                        object_name="container_box",
                        hand_name="right_hand",
                        distance_m=cont_dist,
                        action=HOIAction.APPROACH,
                        duration_frames=1
                    ))

        # 3. Posture ergonomics check
        if pose.rom_limits_violated and primary_activity != "IDLE":
            primary_activity += " [ROM LIMIT]"

        # 4. Real-time Non-Procedural Motion & Activity Recognition
        if primary_activity == "IDLE":
            detected_non_proc = self._detect_non_procedural_motion(pose, cont_pos)
            if detected_non_proc:
                primary_activity = detected_non_proc

        self.previous_wrist_pos = wrist_pos

        # Record action snapshot into temporal sliding window
        self.action_history.append({
            "activity": primary_activity,
            "hoi_actions": [h.action.value for h in active_hoi],
            "lid_angle": lid_angle,
            "objects_inside": {name: obj.is_inside_container for name, obj in objects.items()
                               if name not in self.EXCLUDED_TARGETS},
            "extracted": dict(self.previously_extracted),
        })
        if len(self.action_history) > self.action_history_max:
            self.action_history.pop(0)

        return active_hoi, objects, primary_activity

    def _detect_non_procedural_motion(self, pose: AstronautPose3D, cont_pos: Vector3D) -> Optional[str]:
        """
        Detects any non-procedural human activity from keypoint kinematics when not working on container:
        - Dancing / rhythmic movement
        - Waving / celebrating
        - Clapping
        - Stretching
        - Touching head / face / using phone
        - Hands raised
        - Pointing / gesturing
        - Arms crossed
        - Hands on hips
        - Bending down / crouching
        - Pacing / walking
        - General gesturing / arm movement
        - Standing away
        """
        if not hasattr(pose, "keypoints_2d") or not pose.keypoints_2d:
            return None

        kp = pose.keypoints_2d
        rw = kp.get("right_wrist")
        lw = kp.get("left_wrist")
        r_sh = kp.get("right_shoulder")
        l_sh = kp.get("left_shoulder")
        nose = kp.get("nose")
        r_ear = kp.get("right_ear")
        l_ear = kp.get("left_ear")
        r_hip = kp.get("right_hip")
        l_hip = kp.get("left_hip")
        r_elb = kp.get("right_elbow")
        l_elb = kp.get("left_elbow")

        if not (rw and lw and r_sh and l_sh):
            return None

        # If hand is in close contact with container, operator is working on procedure
        wrist_obj = pose.joints.get("wrist")
        if wrist_obj:
            if wrist_obj.pos_rack.distance_to(cont_pos) <= self.contact_threshold_m + 0.05:
                self.dance_frames = 0
                self.non_proc_frames = 0
                self.active_non_proc_act = None
                return None

        avg_sh_y = (r_sh[1] + l_sh[1]) / 2.0
        avg_sh_x = (r_sh[0] + l_sh[0]) / 2.0
        sh_dist = max(20.0, abs(r_sh[0] - l_sh[0]))

        avg_hip_y = (r_hip[1] + l_hip[1]) / 2.0 if (r_hip and l_hip) else avg_sh_y + sh_dist * 1.6
        avg_hip_x = (r_hip[0] + l_hip[0]) / 2.0 if (r_hip and l_hip) else avg_sh_x

        # Track temporal kinematics
        curr_pt = {
            "rw": (rw[0], rw[1]),
            "lw": (lw[0], lw[1]),
            "rh": (r_hip[0], r_hip[1]) if r_hip else (0, 0),
            "lh": (l_hip[0], l_hip[1]) if l_hip else (0, 0),
            "sh": (avg_sh_x, avg_sh_y)
        }
        self.pose_history.append(curr_pt)
        if len(self.pose_history) > 15:
            self.pose_history.pop(0)

        avg_disp = 0.0
        hip_disp = 0.0
        rw_x_disp = 0.0
        lw_x_disp = 0.0
        if len(self.pose_history) >= 4:
            total_disp = 0.0
            total_hip = 0.0
            total_rw_x = 0.0
            total_lw_x = 0.0
            for i in range(1, len(self.pose_history)):
                p0 = self.pose_history[i - 1]
                p1 = self.pose_history[i]
                d_rw = math.hypot(p1["rw"][0] - p0["rw"][0], p1["rw"][1] - p0["rw"][1])
                d_lw = math.hypot(p1["lw"][0] - p0["lw"][0], p1["lw"][1] - p0["lw"][1])
                d_h = math.hypot(p1["rh"][0] - p0["rh"][0], p1["rh"][1] - p0["rh"][1]) if (p0["rh"][0] > 0 and p1["rh"][0] > 0) else 0.0
                total_disp += d_rw + d_lw + d_h * 1.5
                total_hip += d_h
                total_rw_x += abs(p1["rw"][0] - p0["rw"][0])
                total_lw_x += abs(p1["lw"][0] - p0["lw"][0])
            n_hist = len(self.pose_history) - 1
            avg_disp = total_disp / n_hist
            hip_disp = total_hip / n_hist
            rw_x_disp = total_rw_x / n_hist
            lw_x_disp = total_lw_x / n_hist

        candidate_act: Optional[str] = None

        # 1. Check touching head / face / phone
        d_r_ear = math.hypot(rw[0] - r_ear[0], rw[1] - r_ear[1]) if (r_ear and rw[2] > 0.35) else 999.0
        d_l_ear = math.hypot(lw[0] - l_ear[0], lw[1] - l_ear[1]) if (l_ear and lw[2] > 0.35) else 999.0
        d_r_nose = math.hypot(rw[0] - nose[0], rw[1] - nose[1]) if (nose and rw[2] > 0.35) else 999.0
        d_l_nose = math.hypot(lw[0] - nose[0], lw[1] - nose[1]) if (nose and lw[2] > 0.35) else 999.0

        ear_thresh = max(42.0, sh_dist * 0.42)
        face_thresh = max(42.0, sh_dist * 0.42)

        if min(d_r_nose, d_l_nose) < face_thresh and min(d_r_nose, d_l_nose) < min(d_r_ear, d_l_ear):
            candidate_act = "TOUCHING FACE / HEAD"
        elif min(d_r_ear, d_l_ear) < ear_thresh:
            candidate_act = "USING PHONE"
        elif min(d_r_nose, d_l_nose) < face_thresh:
            candidate_act = "TOUCHING FACE / HEAD"

        # 2. Check wide stretching (both arms extended laterally or wide)
        wrist_span = abs(rw[0] - lw[0])
        both_arms_wide = (rw[0] - r_sh[0] > sh_dist * 0.5) and (l_sh[0] - lw[0] > sh_dist * 0.5)
        if not candidate_act and both_arms_wide and wrist_span > sh_dist * 2.1 and rw[1] > avg_sh_y - 50 and lw[1] > avg_sh_y - 50:
            candidate_act = "STRETCHING"

        # 3. Check arms crossed / folded (right wrist crossed to left side, left wrist crossed to right side)
        wrists_dist = math.hypot(rw[0] - lw[0], rw[1] - lw[1])
        is_crossed = (rw[0] < lw[0] - 5.0) and (avg_sh_y + 15 < rw[1] < avg_hip_y - 15) and (avg_sh_y + 15 < lw[1] < avg_hip_y - 15)
        if not candidate_act and is_crossed and wrists_dist < max(50.0, sh_dist * 0.50):
            candidate_act = "ARMS CROSSED"

        # 4. Check clapping (both wrists meeting uncrossed in front of chest)
        if not candidate_act and wrists_dist < max(35.0, sh_dist * 0.35) and avg_sh_y - 20 < rw[1] < avg_sh_y + 120:
            candidate_act = "CLAPPING"

        # 4. Check hands raised / celebrating vs waving
        r_above = rw[1] < avg_sh_y - 25 or (nose and rw[1] < nose[1])
        l_above = lw[1] < avg_sh_y - 25 or (nose and lw[1] < nose[1])
        if not candidate_act:
            if r_above and l_above:
                candidate_act = "WAVING / CELEBRATING" if (rw_x_disp > 4.0 or lw_x_disp > 4.0 or avg_disp > 8.0) else "HANDS RAISED"
            elif (r_above or l_above) and (rw_x_disp > 3.5 or lw_x_disp > 3.5 or avg_disp > 6.0):
                candidate_act = "WAVING"

        # 5. Check dancing / high dynamic rhythmic movement
        if not candidate_act:
            wrists_elevated = (rw[1] < avg_sh_y + 40 or lw[1] < avg_sh_y + 40)
            if avg_disp > 12.0 and wrists_elevated:
                candidate_act = "DANCING"
            elif avg_disp > 20.0:
                candidate_act = "RAPID MOVEMENT"

        # 6. Check pointing / extended gesturing
        if not candidate_act:
            r_reach_x = abs(rw[0] - r_sh[0])
            l_reach_x = abs(lw[0] - l_sh[0])
            if (r_reach_x > sh_dist * 1.25 and rw[1] < avg_hip_y) or (l_reach_x > sh_dist * 1.25 and lw[1] < avg_hip_y):
                candidate_act = "POINTING / GESTURING"


        # 8. Check hands on hips / waist
        if not candidate_act and r_hip and l_hip and r_elb and l_elb:
            d_rw_hip = math.hypot(rw[0] - r_hip[0], rw[1] - r_hip[1])
            d_lw_hip = math.hypot(lw[0] - l_hip[0], lw[1] - l_hip[1])
            elbow_span = abs(r_elb[0] - l_elb[0])
            if d_rw_hip < sh_dist * 0.8 and d_lw_hip < sh_dist * 0.8 and elbow_span > sh_dist * 1.4:
                candidate_act = "HANDS ON HIPS"

        # 9. Check bending down / crouching
        if not candidate_act and nose and r_hip and l_hip:
            torso_height = avg_hip_y - nose[1]
            if torso_height < sh_dist * 0.85:
                candidate_act = "BENDING DOWN"

        # 10. Check pacing / walking
        if not candidate_act and hip_disp > 6.0:
            candidate_act = "PACING / WALKING"

        # 11. Check general non-procedural hand movement / gesturing
        if not candidate_act and avg_disp > 4.5:
            candidate_act = "GESTURING"

        # 12. Check standing away from container
        if not candidate_act and wrist_obj:
            cont_dist = wrist_obj.pos_rack.distance_to(cont_pos)
            if cont_dist > self.approach_threshold_m + 0.15:
                candidate_act = "STANDING AWAY"

        # Filter / debounce activity state
        if candidate_act:
            if candidate_act == self.active_non_proc_act:
                self.non_proc_frames = min(20, self.non_proc_frames + 2)
            else:
                self.active_non_proc_act = candidate_act
                self.non_proc_frames = 2
            if self.non_proc_frames >= 2:
                return candidate_act
        else:
            self.non_proc_frames = max(0, self.non_proc_frames - 1)
            if self.non_proc_frames == 0:
                self.active_non_proc_act = None

        return None


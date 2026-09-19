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

        # Temporal action sequence buffer (sliding window for sequence classifier)
        self.action_history: List[Dict[str, Any]] = []
        self.action_history_max: int = 30  # ~1 second @ 30fps

    def reset(self):
        """Resets all HOI contact counters for a new test cycle."""
        for k in self.contact_frame_counters:
            self.contact_frame_counters[k] = 0
        self.previously_extracted.clear()
        self.previous_wrist_pos = None
        self.last_lid_angle = 0.0
        self.action_history.clear()

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


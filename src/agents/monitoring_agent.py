"""
BAS Autonomous HAR System - Agent 8: Monitoring Agent
Coordinates mission outputs: Dual Video Pipeline (Local NVMe + RTSP IP Streaming),
Offline Speech Synthesis (TTS), Structured JSONL Telemetry, and Live Video Overlays.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from src.core.types import (
    AstronautPose3D,
    ExperimentObject,
    FSMStep,
    AnomalyType,
    EntityState
)
from src.audio.offline_tts import OfflineTTS
from src.telemetry.jsonl_logger import JSONLTelemetryLogger
from src.telemetry.action_session_logger import ActionSessionLogger
from src.telemetry.csv_logger import RealtimeCSVTelemetryLogger
from src.streaming.video_pipeline import DualVideoPipeline


class MonitoringAgent:
    """Master Egress and Mission Telemetry Coordinator."""

    def __init__(
        self,
        enable_tts: bool = True,
        enable_streaming: bool = True,
        stream_port: int = 8080,
        output_video_path: Optional[str] = None,
        output_telemetry_path: Optional[str] = None,
        realtime_feed_dir: str = "realtime_feed",
        output_csv_dir: str = "experiments"
    ):
        self.tts = OfflineTTS() if enable_tts else None
        self.telemetry = JSONLTelemetryLogger(output_telemetry_path)
        self.csv_logger = RealtimeCSVTelemetryLogger(
            output_dir=output_csv_dir,
            realtime_feed_dir=realtime_feed_dir
        )
        self.action_logger = ActionSessionLogger()
        self.video_pipeline = DualVideoPipeline(
            local_output_path=output_video_path,
            stream_port=stream_port
        ) if enable_streaming else None

        self.last_step_logged: Optional[FSMStep] = None

    def reset(self):
        """Resets action session logger and telemetry for new experiment run."""
        if self.action_logger:
            self.action_logger.reset()
        if self.tts:
            self.tts.reset()

    def switch_output_dir(self, output_dir: str, realtime_feed_dir: str = "realtime_feed"):
        """Switches output directories for isolated telemetry logging."""
        if self.csv_logger:
            self.csv_logger.close()
        self.csv_logger = RealtimeCSVTelemetryLogger(
            output_dir=output_dir,
            realtime_feed_dir=realtime_feed_dir
        )

    def check_reset_requested(self) -> bool:
        """Checks if web client requested an experiment reset."""
        if self.video_pipeline:
            return self.video_pipeline.check_and_clear_reset()
        return False

    def process_egress(
        self,
        raw_frame: np.ndarray,
        fused_pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        current_step: FSMStep,
        debounce_count: int,
        anomaly: AnomalyType,
        instruction: str,
        voice_alert: Optional[str],
        transition_event: Optional[str],
        frame_id: int,
        fps: float,
        latency_ms: float,
        twin_canvas: Optional[np.ndarray] = None,
        current_activity: str = "IDLE",
        llm_verification: Optional[Dict[str, Any]] = None,
        source_type: str = "LIVE_WEBCAM",
        is_step_correct: bool = True,
        step_verdict: str = "CORRECT (NOMINAL)",
        experiment_id: str = "BAS-EXP-BOX-RETURN",
        scene_graph: Optional[Dict[str, Any]] = None,
        spatial_relations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Synthesizes audio alerts, logs session actions to JSON, and renders live HUD overlays.
        Returns the annotated display frame.
        """
        # 1. Trigger Voice Alerts / Spoken Guidance
        if voice_alert and self.tts:
            is_priority = (anomaly != AnomalyType.NONE) or (not is_step_correct) or ("Wrong move" in voice_alert)
            self.tts.speak(voice_alert, priority=is_priority)

        # 2. Update Structured Action Session Logger (What I'm doing vs What I have to do)
        timestamp_sec = frame_id / max(1.0, fps)
        self.action_logger.update(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            step=int(current_step),
            step_name=current_step.name,
            what_i_am_doing=current_activity,
            what_i_have_to_do=instruction,
            lid_angle=lid_angle,
            anomaly=anomaly.value
        )

        # 3. Emit Frame-by-Frame CSV Telemetry for 3D & Digital Twin
        self.csv_logger.log_frame(
            frame_id=frame_id,
            fps=fps,
            step=current_step,
            activity=current_activity,
            anomaly=anomaly,
            instruction=instruction,
            lid_angle=lid_angle,
            pose=fused_pose,
            objects=objects,
            llm_verification=llm_verification
        )

        # 4. Emit Milestone Telemetry Records
        if transition_event is not None:
            self.telemetry.log_event(
                frame_id=frame_id,
                state_id=int(current_step),
                state_name=current_step.name,
                event_name=transition_event,
                status="SUCCESS",
                anomaly=anomaly.value,
                tts_prompt=instruction,
                details={"lid_angle": round(lid_angle, 1), "activity": current_activity}
            )
        elif anomaly != AnomalyType.NONE:
            self.telemetry.log_event(
                frame_id=frame_id,
                state_id=int(current_step),
                state_name=current_step.name,
                event_name="ANOMALY_DETECTED",
                status="ANOMALY",
                anomaly=anomaly.value,
                tts_prompt=instruction
            )

        # Periodic Heartbeat
        self.telemetry.log_heartbeat(frame_id, int(current_step), current_step.name, fps, latency_ms)

        # 5. Composite Live Stream HUD Overlay
        annotated_frame = self._draw_hud(
            raw_frame=raw_frame,
            pose=fused_pose,
            objects=objects,
            lid_angle=lid_angle,
            current_step=current_step,
            debounce_count=debounce_count,
            anomaly=anomaly,
            instruction=instruction,
            fps=fps,
            twin_canvas=twin_canvas,
            current_activity=current_activity,
            llm_verification=llm_verification,
            source_type=source_type,
            is_step_correct=is_step_correct,
            step_verdict=step_verdict,
            experiment_id=experiment_id,
            spatial_relations=spatial_relations
        )

        # 6. Dispatch to Dual Video Pipeline (Local MP4 + RTSP Stream + Web API)
        if self.video_pipeline:
            self.video_pipeline.set_active_source_type(source_type)
            self.video_pipeline.set_active_experiment_id(experiment_id)
            self.video_pipeline.write_frame(annotated_frame, telemetry={
                "step": int(current_step),
                "step_name": current_step.name,
                "debounce": debounce_count,
                "debounce_max": 6,
                "anomaly": anomaly.value,
                "instruction": instruction,
                "transition_event": transition_event,
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "lid_angle": round(lid_angle, 1),
                "frame_id": frame_id,
                "what_i_am_doing": current_activity,
                "what_i_have_to_do": instruction,
                "source_type": source_type,
                "is_step_correct": is_step_correct,
                "step_verdict": step_verdict,
                "experiment_id": experiment_id
            }, scene_graph=scene_graph)

        return annotated_frame

    def check_source_switch_requested(self) -> Optional[str]:
        """Returns requested source ('0' or 'clip1.mp4') if switched via UI, else None."""
        if self.video_pipeline:
            return self.video_pipeline.check_and_clear_source_switch()
        return None

    def check_experiment_switch_requested(self) -> Optional[str]:
        """Returns requested protocol path if switched via UI, else None."""
        if self.video_pipeline:
            return self.video_pipeline.check_and_clear_experiment_switch()
        return None

    def _draw_hud(
        self,
        raw_frame: np.ndarray,
        pose: AstronautPose3D,
        objects: Dict[str, ExperimentObject],
        lid_angle: float,
        current_step: FSMStep,
        debounce_count: int,
        anomaly: AnomalyType,
        instruction: str,
        fps: float,
        twin_canvas: Optional[np.ndarray] = None,
        current_activity: str = "IDLE",
        llm_verification: Optional[Dict[str, Any]] = None,
        source_type: str = "LIVE_WEBCAM",
        is_step_correct: bool = True,
        step_verdict: str = "CORRECT (NOMINAL)",
        experiment_id: str = "BAS-EXP-BOX-RETURN",
        spatial_relations: Optional[List[str]] = None
    ) -> np.ndarray:
        frame = raw_frame.copy()
        h, w, _ = frame.shape

        # Adaptive layout scaling based on resolution
        scale = max(0.40, min(0.70, w / 1280.0))
        thick = 1 if w < 1000 else 2
        banner_h = max(34, min(50, int(h * 0.065)))
        bot_h = max(32, min(48, int(h * 0.060)))

        # 1. Draw 2D Object Bounding Boxes & Staggered Badges (Zero Overlap)
        color_map = {
            "container_box": (160, 160, 160),
            "container_lid": (0, 240, 200),
            "component_box": (255, 140, 0),
            "red_box": (60, 70, 255),
            "yellow_box": (0, 230, 255),
            "human_body": (0, 215, 255),
            "person": (0, 215, 255),
            "cell phone": (0, 255, 128),
            "bottle": (255, 105, 180),
            "cup": (255, 140, 0),
            "book": (186, 85, 211),
            "laptop": (0, 191, 255),
            "scissors": (255, 20, 147),
            "mouse": (50, 205, 50),
            "chair": (169, 169, 169)
        }

        def get_obj_color(obj_name: str) -> Tuple[int, int, int]:
            base = obj_name.split("_")[0]
            if base in color_map:
                return color_map[base]
            h_val = (abs(hash(obj_name)) * 37) % 180
            hsv_pix = np.array([[[h_val, 220, 240]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)[0][0]
            return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        # Strict Filter: ONLY label experiment things on screen (container, lid, component payload, hands, human body)
        # Suppress arbitrary background clutter (chair, laptop, desk, etc.)
        EXPERIMENT_ALLOWED_NAMES = {
            "container_box", "container_lid", "component_box",
            "red_box", "yellow_box", "astronaut_hand", "operator_hand", "human_body", "person"
        }

        occupied_badge_rects: List[Tuple[int, int, int, int]] = []

        for name, obj in objects.items():
            is_experiment_thing = (
                name in EXPERIMENT_ALLOWED_NAMES
                or any(k in name for k in ("container", "box", "lid", "component", "hand", "human", "person"))
            )
            if not is_experiment_thing:
                continue

            if obj.bbox:
                bx1, by1 = int(obj.bbox.xmin), int(obj.bbox.ymin)
                bx2, by2 = int(obj.bbox.xmax), int(obj.bbox.ymax)
                col = get_obj_color(name)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, max(1, int(1.5 * scale * 2)))

                # Text format: clean experiment labels
                if "container_lid" in name:
                    status_txt = f"CONTAINER LID [{lid_angle:.0f}°]"
                elif "container_box" in name:
                    status_txt = "CONTAINER BOX"
                elif "component_box" in name:
                    c_tag = f" ({obj.class_name.upper()})" if obj.class_name and obj.class_name != "component_box" else ""
                    status_txt = f"COMPONENT{c_tag} [{obj.state.value}]"
                elif "human_body" in name:
                    c_conf = f" {int(obj.bbox.confidence * 100)}%" if obj.bbox and obj.bbox.confidence else ""
                    status_txt = f"HUMAN BODY{c_conf}"
                elif "red_box" in name:
                    status_txt = f"RED BOX [{obj.state.value}]"
                elif "yellow_box" in name:
                    status_txt = f"YELLOW BOX [{obj.state.value}]"
                elif obj.bbox.confidence and obj.bbox.confidence > 0.05:
                    status_txt = f"{obj.name.upper()} {int(obj.bbox.confidence * 100)}%"
                else:
                    status_txt = f"{obj.name.upper()} [{obj.state.value}]"
                font_scale = max(0.34, scale * 0.85)
                (tw, th), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                pad_x, pad_y = 6, 4
                badge_w = tw + pad_x * 2
                badge_h = th + pad_y * 2

                # Target position: Above the box if there is room below top banner, otherwise inside
                rx1 = max(4, min(w - badge_w - 4, bx1))
                if by1 - badge_h - 2 >= banner_h + 2:
                    ry1 = by1 - badge_h - 2
                else:
                    ry1 = min(h - bot_h - badge_h - 2, by1 + 4)

                # Collision resolution with previously placed badges
                for (ox1, oy1, ox2, oy2) in occupied_badge_rects:
                    if not (rx1 + badge_w < ox1 or rx1 > ox2 or ry1 + badge_h < oy1 or ry1 > oy2):
                        ry1 = min(h - bot_h - badge_h - 2, oy2 + 3)

                rx2 = rx1 + badge_w
                ry2 = ry1 + badge_h
                occupied_badge_rects.append((rx1, ry1, rx2, ry2))

                # Draw badge background and high-contrast text
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, -1)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (10, 10, 10), 1)
                cv2.putText(
                    frame, status_txt,
                    (rx1 + pad_x, ry1 + th + pad_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 1, cv2.LINE_AA
                )

                # Draw relationship tags if they exist
                if hasattr(obj, "relations") and obj.relations:
                    rel_y = ry2 + 3
                    for rel in obj.relations[:2]: # Show top 2 relations to avoid UI clutter
                        rel_txt = f"{rel.predicate} {rel.object_name}"
                        (rtw, rth), _ = cv2.getTextSize(rel_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, 1)
                        rbadge_w = rtw + pad_x * 2
                        rbadge_h = rth + pad_y * 2
                        cv2.rectangle(frame, (rx1, rel_y), (rx1 + rbadge_w, rel_y + rbadge_h), col, -1)
                        cv2.rectangle(frame, (rx1, rel_y), (rx1 + rbadge_w, rel_y + rbadge_h), (10, 10, 10), 1)
                        cv2.putText(
                            frame, rel_txt,
                            (rx1 + pad_x, rel_y + rth + pad_y - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (10, 10, 10), 1, cv2.LINE_AA
                        )
                        occupied_badge_rects.append((rx1, rel_y, rx1 + rbadge_w, rel_y + rbadge_h))
                        rel_y += rbadge_h + 3

        # 2. Draw Full Person Skeleton & Biomechanical Joints
        if hasattr(pose, "keypoints_2d") and pose.keypoints_2d:
            kp_dict = pose.keypoints_2d

            # 17 COCO Bone Connections with specialized anatomical color-coding
            BONE_CONNECTIONS = [
                # Head / Face (warm gold)
                ("nose", "left_eye", (0, 215, 255)),
                ("nose", "right_eye", (0, 215, 255)),
                ("left_eye", "left_ear", (0, 190, 240)),
                ("right_eye", "right_ear", (0, 190, 240)),
                # Shoulders & Torso (emerald green)
                ("left_shoulder", "right_shoulder", (0, 255, 128)),
                ("left_shoulder", "left_hip", (0, 240, 100)),
                ("right_shoulder", "right_hip", (0, 240, 100)),
                ("left_hip", "right_hip", (0, 255, 128)),
                # Upper Limbs (Left Arm = Cyan, Right Arm = Amber/Orange)
                ("left_shoulder", "left_elbow", (0, 230, 255)),
                ("left_elbow", "left_wrist", (0, 255, 255)),
                ("right_shoulder", "right_elbow", (255, 190, 0)),
                ("right_elbow", "right_wrist", (255, 220, 0)),
                # Lower Limbs (Pastel Violet / Cyan)
                ("left_hip", "left_knee", (255, 140, 220)),
                ("left_knee", "left_ankle", (255, 140, 220)),
                ("right_hip", "right_knee", (220, 140, 255)),
                ("right_knee", "right_ankle", (220, 140, 255)),
            ]

            # Render bone segments with high-contrast borders
            line_w = max(2, int(2.5 * scale * 2))
            for pt1_name, pt2_name, bone_col in BONE_CONNECTIONS:
                if pt1_name in kp_dict and pt2_name in kp_dict:
                    x1, y1, c1 = kp_dict[pt1_name]
                    x2, y2, c2 = kp_dict[pt2_name]
                    if c1 > 0.20 and c2 > 0.20:
                        p1 = (int(x1), int(y1))
                        p2 = (int(x2), int(y2))
                        # Black drop shadow for maximum visibility
                        cv2.line(frame, p1, p2, (15, 15, 15), line_w + 2, cv2.LINE_AA)
                        # Main vibrant bone link
                        cv2.line(frame, p1, p2, bone_col, line_w, cv2.LINE_AA)

            # Render anatomical joint nodes
            for j_name, (jx, jy, jconf) in kp_dict.items():
                if jconf > 0.20:
                    p = (int(jx), int(jy))
                    if "wrist" in j_name:
                        # Prominent active operator hand target ring
                        cv2.circle(frame, p, max(7, int(9 * scale)), (10, 10, 10), -1)
                        cv2.circle(frame, p, max(5, int(7 * scale)), (0, 255, 255), -1)
                        cv2.circle(frame, p, max(3, int(4 * scale)), (255, 255, 255), -1)
                        cv2.circle(frame, p, max(9, int(12 * scale)), (0, 255, 255), 1, cv2.LINE_AA)
                    elif "elbow" in j_name or "shoulder" in j_name:
                        # Major limb joint
                        cv2.circle(frame, p, max(5, int(6 * scale)), (10, 10, 10), -1)
                        cv2.circle(frame, p, max(4, int(5 * scale)), (0, 240, 120), -1)
                        cv2.circle(frame, p, max(2, int(3 * scale)), (255, 255, 255), -1)
                    else:
                        # Standard joint node
                        cv2.circle(frame, p, max(4, int(5 * scale)), (10, 10, 10), -1)
                        cv2.circle(frame, p, max(3, int(4 * scale)), (255, 255, 255), -1)

            # Label dominant active wrist pill badge
            if "wrist" in kp_dict and kp_dict["wrist"][2] > 0.20:
                wx, wy, _ = kp_dict["wrist"]
                wx_i, wy_i = int(wx), int(wy)
                w_txt = f"ASTRONAUT WRIST [{pose.elbow_angle_deg:.0f}°]"
                w_scale = max(0.32, scale * 0.75)
                (wtw, wth), _ = cv2.getTextSize(w_txt, cv2.FONT_HERSHEY_SIMPLEX, w_scale, 1)
                wrx1 = max(4, min(w - wtw - 10, wx_i + 12))
                wry1 = max(banner_h + 2, min(h - bot_h - wth - 8, wy_i - wth // 2 - 3))
                wrx2 = wrx1 + wtw + 8
                wry2 = wry1 + wth + 6

                for (ox1, oy1, ox2, oy2) in occupied_badge_rects:
                    if not (wrx1 + (wrx2 - wrx1) < ox1 or wrx1 > ox2 or wry1 + (wry2 - wry1) < oy1 or wry1 > oy2):
                        wry1 = min(h - bot_h - (wry2 - wry1) - 2, oy2 + 3)
                        wry2 = wry1 + wth + 6

                cv2.rectangle(frame, (wrx1, wry1), (wrx2, wry2), (20, 20, 20), -1)
                cv2.rectangle(frame, (wrx1, wry1), (wrx2, wry2), (0, 240, 255), 1)
                cv2.putText(
                    frame, w_txt,
                    (wrx1 + 4, wry1 + wth + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, w_scale, (0, 240, 255), 1, cv2.LINE_AA
                )
        elif pose.joints.get("wrist"):
            # Fallback for single wrist extrapolation
            wrist = pose.joints.get("wrist")
            if wrist:
                wx = int(self.video_pipeline.resolution[0]/2 + wrist.pos_camera.x * 400) if self.video_pipeline else w//2
                wy = int(self.video_pipeline.resolution[1]/2 + wrist.pos_camera.y * 400) if self.video_pipeline else h//2
                wx = max(15, min(w - 15, wx))
                wy = max(banner_h + 15, min(h - bot_h - 15, wy))

                cv2.circle(frame, (wx, wy), max(5, int(8 * scale)), (0, 240, 255), -1)
                cv2.circle(frame, (wx, wy), max(7, int(11 * scale)), (255, 255, 255), 1)

        # 3. Streamlined Minimalist HUD Overlay (Header & Footer)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (8, 12, 18), -1)
        cv2.rectangle(overlay, (0, h - bot_h), (w, h), (8, 12, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Separator lines
        is_dual_experiment = (
            "RED" in experiment_id.upper()
            or "26174" in experiment_id
            or "RED_YELLOW" in source_type.upper()
        )
        is_complete = (
            current_step in (FSMStep.COMPLETE, FSMStep.BOX_CLOSED)
            or int(current_step) >= (5 if is_dual_experiment else 4)
        )
        sep_col = (0, 230, 120) if is_complete else ((0, 60, 220) if anomaly != AnomalyType.NONE else (0, 180, 230))
        cv2.line(frame, (0, banner_h), (w, banner_h), sep_col, 1)
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), sep_col, 1)

        # Header Zone 1 (Left): Mission Identifier & Real-Time Source Indicator
        if source_type == "LIVE_WEBCAM":
            src_tag = "LIVE #0"
            title_col = (0, 240, 150)
        elif "RED_YELLOW" in source_type:
            src_tag = "RED-YELLOW"
            title_col = (50, 220, 255)
        else:
            src_tag = "DEMO CLIP"
            title_col = (255, 220, 100)
        title_txt = f"BAS HAR | {src_tag}"
        title_scale = max(0.38, scale * 0.90)
        (tw, th), _ = cv2.getTextSize(title_txt, cv2.FONT_HERSHEY_SIMPLEX, title_scale, thick)
        title_y = (banner_h + th) // 2
        cv2.putText(frame, title_txt, (14, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_col, thick, cv2.LINE_AA)
        left_bound = 14 + tw + 14

        # Header Zone 3 (Right): Telemetry Status
        status_scale = max(0.34, scale * 0.82)
        llm_s_tag = ""
        if llm_verification:
            llm_v_step = llm_verification.get("verified_step", int(current_step))
            llm_s_tag = f" | VLM: S{llm_v_step}"
        status_txt = f"FPS: {fps:.0f} | LID: {lid_angle:.0f}d | DEB: {debounce_count}/6{llm_s_tag}"
        (sw, sh), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)
        status_x = max(left_bound + 10, w - sw - 14)
        status_y = (banner_h + sh) // 2
        cv2.putText(frame, status_txt, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, status_scale, (180, 215, 235), 1, cv2.LINE_AA)
        right_bound = status_x - 14

        # Header Zone 2 (Center): Active Procedure Step (Guaranteed No Overlap)
        if is_dual_experiment:
            step_labels = {
                0: "S0: STANDBY",
                1: "S1: CONTAINER OPEN",
                2: "S2: RED EXTRACTED",
                3: "S3: YELLOW EXTRACTED",
                4: "S4: OBJECTS RETURNED",
                5: "S5: BOX CLOSED"
            }
        else:
            step_labels = {
                0: "S0: STANDBY",
                1: "S1: CONTAINER OPEN",
                2: "S2: OBJECT EXTRACTED",
                3: "S3: OBJECT RETURNED",
                4: "S4: MISSION COMPLETE",
                5: "S5: BOX CLOSED"
            }
        active_step_txt = step_labels.get(int(current_step), current_step.name)
        step_scale = max(0.36, scale * 0.86)
        (step_w, step_h), _ = cv2.getTextSize(active_step_txt, cv2.FONT_HERSHEY_SIMPLEX, step_scale, 1)

        # Calculate center position & clamp between left and right bounds
        ideal_center_x = (w - step_w) // 2
        center_x = max(left_bound, min(right_bound - step_w, ideal_center_x))
        if center_x + step_w <= right_bound:
            step_badge_col = (0, 230, 120) if is_complete else (0, 210, 255)
            pill_pad_x = 8
            pill_pad_y = 4
            px1 = center_x - pill_pad_x
            py1 = max(2, (banner_h - (step_h + pill_pad_y * 2)) // 2)
            px2 = center_x + step_w + pill_pad_x
            py2 = min(banner_h - 2, py1 + step_h + pill_pad_y * 2)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (20, 30, 45), -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), step_badge_col, 1)
            cv2.putText(
                frame, active_step_txt,
                (center_x, (banner_h + step_h) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, step_scale, step_badge_col, 1, cv2.LINE_AA
            )

        # Footer: Action Bar & Step Verification Badge
        act_scale = max(0.36, scale * 0.85)
        act_display = current_activity if current_activity != "IDLE" else "OBSERVING"
        doing_txt = f"DOING: {act_display}"
        (dw, dh), _ = cv2.getTextSize(doing_txt, cv2.FONT_HERSHEY_SIMPLEX, act_scale, 1)
        bot_y = h - (bot_h - dh) // 2 - 2

        cv2.rectangle(frame, (10, h - bot_h + 4), (10 + dw + 12, h - 4), (20, 35, 50), -1)
        cv2.rectangle(frame, (10, h - bot_h + 4), (10 + dw + 12, h - 4), (0, 220, 255), 1)
        cv2.putText(frame, doing_txt, (16, bot_y), cv2.FONT_HERSHEY_SIMPLEX, act_scale, (0, 230, 255), 1, cv2.LINE_AA)

        doing_end = 10 + dw + 18

        # Real-time Step Correctness Pill
        verd_txt = "STEP: OK" if is_step_correct else "WRONG STEP!"
        verd_col = (0, 230, 120) if is_step_correct else (20, 30, 255)
        (vw, vh), _ = cv2.getTextSize(verd_txt, cv2.FONT_HERSHEY_SIMPLEX, act_scale, 2 if not is_step_correct else 1)
        vx1 = doing_end
        cv2.rectangle(frame, (vx1, h - bot_h + 4), (vx1 + vw + 14, h - 4), (15, 30, 25) if is_step_correct else (20, 20, 180), -1)
        cv2.rectangle(frame, (vx1, h - bot_h + 4), (vx1 + vw + 14, h - 4), (255, 255, 255) if not is_step_correct else verd_col, 2 if not is_step_correct else 1)
        cv2.putText(frame, verd_txt, (vx1 + 6, bot_y), cv2.FONT_HERSHEY_SIMPLEX, act_scale, (255, 255, 255) if not is_step_correct else verd_col, 2 if not is_step_correct else 1, cv2.LINE_AA)

        guide_start = vx1 + vw + 20

        # Guidance / Alert Text on the right with safe truncation
        guide_col = (60, 90, 255) if (anomaly != AnomalyType.NONE or not is_step_correct) else (255, 255, 255)
        prefix = "ALERT: " if (anomaly != AnomalyType.NONE or not is_step_correct) else "NEXT: "
        full_guide = f"{prefix}{instruction}"

        guide_scale = max(0.34, scale * 0.82)
        avail_width = w - guide_start - 16

        (gw, gh), _ = cv2.getTextSize(full_guide, cv2.FONT_HERSHEY_SIMPLEX, guide_scale, 1)
        display_guide = full_guide
        if gw > avail_width and avail_width > 60:
            while len(display_guide) > 8:
                display_guide = display_guide[:-4] + "..."
                (gw, gh), _ = cv2.getTextSize(display_guide, cv2.FONT_HERSHEY_SIMPLEX, guide_scale, 1)
                if gw <= avail_width:
                    break

        if avail_width > 40:
            cv2.putText(frame, display_guide, (guide_start, bot_y), cv2.FONT_HERSHEY_SIMPLEX, guide_scale, guide_col, 1, cv2.LINE_AA)

        # Prominent Central Alert Banner for WRONG STEP (Temporarily Disabled by User Request)
        # if anomaly != AnomalyType.NONE or not is_step_correct:
        #     warn_txt = f"WRONG STEP DETECTED: {instruction.upper()}"
        #     w_scale = max(0.38, scale * 0.95)
        #     (wtw, wth), _ = cv2.getTextSize(warn_txt, cv2.FONT_HERSHEY_SIMPLEX, w_scale, 2)
        #     wx1 = max(10, (w - wtw) // 2 - 14)
        #     wy1 = banner_h + 8
        #     wx2 = min(w - 10, wx1 + wtw + 28)
        #     wy2 = wy1 + wth + 14
        #     cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (15, 15, 200), -1)
        #     cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (255, 255, 255), 2)
        #     cv2.putText(frame, warn_txt, (wx1 + 14, wy1 + wth + 7), cv2.FONT_HERSHEY_SIMPLEX, w_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Composite 3D Digital Twin Canvas (Picture-in-Picture)
        if twin_canvas is not None:
            tc_h, tc_w = twin_canvas.shape[:2]
            pip_w, pip_h = 320, 240
            tc_resized = cv2.resize(twin_canvas, (pip_w, pip_h))
            pip_x = w - pip_w - 10
            pip_y = banner_h + 10
            # Draw border
            cv2.rectangle(frame, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (255, 255, 255), 2)
            frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = tc_resized

        # Display Spatial Relations (Always show the section to indicate it's active)
        rel_scale = max(0.32, scale * 0.8)
        start_y = banner_h + 20
        start_x = 10
        
        # Draw background panel for readability
        panel_h = 20 + max(1, min(5, len(spatial_relations) if spatial_relations else 1)) * 20
        panel_w = max(200, int(w * 0.25))
        cv2.rectangle(frame, (start_x - 5, banner_h + 5), (start_x + panel_w, banner_h + 5 + panel_h), (20, 20, 30), -1)
        cv2.rectangle(frame, (start_x - 5, banner_h + 5), (start_x + panel_w, banner_h + 5 + panel_h), (0, 255, 255), 1)
        
        cv2.putText(frame, "SPATIAL RELATIONS:", (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, rel_scale, (0, 255, 255), 1, cv2.LINE_AA)
        
        if spatial_relations:
            for i, rel_str in enumerate(spatial_relations[:5]): # Show top 5
                y_offset = start_y + (i + 1) * 20
                cv2.putText(frame, rel_str, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, rel_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Searching...", (start_x, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, rel_scale, (150, 150, 150), 1, cv2.LINE_AA)

        return frame

    def close(self):
        if self.action_logger:
            self.action_logger.save()
        if self.video_pipeline:
            self.video_pipeline.close()
        if self.telemetry:
            self.telemetry.close()
        if self.csv_logger:
            self.csv_logger.close()
        if self.tts:
            self.tts.shutdown()

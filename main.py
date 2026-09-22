"""
BAS Autonomous HAR System - Master Multi-Agent Orchestrator
Bridges all 8 specialized agents over the Digital Twin Memory Blackboard,
providing real-time procedural tracking, deterministic validation, voice alerts,
and dual-stream video output for the Bharatiya Antariksh Station (BAS).
"""

import sys
import os

# Auto-switch to project virtual environment interpreter if launched via global python
_workspace_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_workspace_dir, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([_venv_python] + sys.argv))

_venv_site = os.path.join(_workspace_dir, ".venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import time
import argparse
import cv2
import numpy as np

# Add RelateAnything to path
sys.path.insert(0, os.path.join(_workspace_dir, "RelateAnything-main"))
try:
    from deploy.postprocess import ThresholdConfig # type: ignore
    from deploy.runtime import DetectorConfig, ScenePipeline # type: ignore
except ImportError as e:
    print(f"CRITICAL: Failed to import ScenePipeline: {e}")
    ScenePipeline = None

# Core & Blackboard
from src.core.types import FSMStep, AnomalyType
from src.core.shared_memory import DigitalTwinBlackboard

# 8 Specialized Agents + Real-Time Local LLM Verifier
from src.agents.perception_agent import PerceptionAgent
from src.agents.imu_agent import IMUAgent
from src.agents.fusion_agent import FusionAgent
from src.agents.har_agent import HARAgent
from src.agents.spatial_agent import SpatialAgent
from src.agents.digital_twin_agent import DigitalTwinAgent
from src.agents.validation_agent import ValidationAgent

from src.agents.reasoning_agent import ReasoningAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.llm.realtime_llm_verifier import RealtimeLLMVerifier


def run_orchestrator(
    source="auto",
    config_path=None,
    use_desktop_gui=False,
    show_window=True,
    enable_tts=True,
    enable_streaming=True,
    stream_port=8080,
    max_frames=None,
    realtime_feed_dir="realtime_feed",
    output_csv_dir=None,
    record_har_dataset=False
):
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - ON-BOARD HAR MULTI-AGENT SYSTEM")
    print("   Autonomous Sequence Validation & Digital Twin Architecture")
    print("   ISRO SIH Problem Statement #26174")
    print("=" * 70)

    # Detect if source or protocol is the Red-Yellow experiment
    is_red_yellow = (
        "red_yellow" in str(source).lower()
        or "clip.mp4" in str(source).lower()
        or (config_path is not None and "red_yellow" in str(config_path).lower())
    )

    if is_red_yellow:
        if config_path is None or "box_return" in str(config_path):
            config_path = "configs/red_yellow_fsm.json"
        if realtime_feed_dir == "realtime_feed":
            realtime_feed_dir = "realtime_feed_red_yellow"
        if output_csv_dir is None:
            output_csv_dir = "experiments_red_yellow"
    else:
        if config_path is None:
            config_path = "configs/box_return_fsm.json"
        if output_csv_dir is None:
            output_csv_dir = "experiments"

    print(f"[Orchestrator] Active Procedure Protocol: {config_path}")
    print(f"[Orchestrator] Dedicated Real-Time Feed Directory: '{realtime_feed_dir}/'")
    print(f"[Orchestrator] Dedicated Telemetry Output Directory: '{output_csv_dir}/'")

    # 1. Initialize Shared Memory Blackboard
    blackboard = DigitalTwinBlackboard()

    # 1.5. Initialize RelateAnything ScenePipeline
    print("[Orchestrator] Initializing RelateAnything ScenePipeline for Spatial Relations...")
    relate_pipe = None
    if ScenePipeline is not None:
        try:
            relate_pipe = ScenePipeline(
                dist_dir="RelateAnything-main/deploy/dist/relsgg-vits16plus",
                detector="RelateAnything-main/deploy/dist/detector-local/detector.onnx",
                backend="onnx",
                det_cfg=DetectorConfig(conf=0.25, iou=0.5, max_det=32),
                thr_cfg=ThresholdConfig(threshold=0.5, topk=20)
            )
            try:
                relate_pipe.rel.set_predicates([
                    "on", "on top of", "in front of", "behind", "beside", 
                    "inside", "contained in", "above", "below", "holding"
                ])
            except Exception:
                pass # Use baked predicates
        except Exception as e:
            print(f"[Orchestrator] Warning: Could not init RelateAnything pipeline: {e}")

    # 2. Instantiate Real-Time Asynchronous Local VLM Verifier (Ollama Qwen3-VL)
    print("[Orchestrator] Engaging Multimodal VLM Real-Time Verifier (qwen3-vl:2b-instruct @ http://localhost:11434)...")
    llm_verifier = RealtimeLLMVerifier(
        model_name="qwen3-vl:2b-instruct",
        experiment_id="BAS-EXP-RED-YELLOW" if is_red_yellow else "BAS-EXP-BOX-RETURN"
    )

    # 3. Instantiate the 8 Specialized Agents
    print("[Orchestrator] Initializing 8 Specialized Agents...")
    agent_perception = PerceptionAgent()
    agent_imu = IMUAgent()
    agent_fusion = FusionAgent()
    agent_spatial = SpatialAgent()
    agent_har = HARAgent()
    agent_twin = DigitalTwinAgent(is_dual=is_red_yellow)
    print("[Orchestrator] Engaging Validation Engine...")
    agent_validation = ValidationAgent(config_path=config_path)
    agent_reasoning = ReasoningAgent(config_path=config_path)
    agent_monitoring = MonitoringAgent(
        enable_tts=enable_tts,
        enable_streaming=enable_streaming,
        stream_port=stream_port,
        realtime_feed_dir=realtime_feed_dir,
        output_csv_dir=output_csv_dir
    )
    llm_verifier.set_protocol(agent_validation.experiment_id)

    def reset_pipeline():
        """Cleanly resets all 8 agents, LLM verifier, and blackboard for a fresh run."""
        agent_perception.reset()
        agent_imu.reset()
        agent_fusion.reset()
        agent_har.reset()
        agent_validation.reset()
        agent_reasoning.reset()
        agent_monitoring.reset()
        llm_verifier.reset()
        blackboard.reset()

    # Optional Desktop GUI
    desktop_gui = None
    if use_desktop_gui:
        try:
            from src.gui.mission_gui import MissionControlGUI
            desktop_gui = MissionControlGUI()
            # Suppress duplicate floating OpenCV window to ensure only 1 unified window opens
            show_window = False
            print("[Orchestrator] Desktop Mission Control GUI initialized (single unified window).")
        except Exception as e:
            print(f"[Orchestrator] Desktop GUI warning: {e}. Falling back to Web/OpenCV mode.")
            show_window = True

    # 3. Setup Video Source (Auto-Probe Live Webcam first for Real-Time Detection)
    source_type = "LIVE_WEBCAM"
    cap = None

    def open_hardware_camera(idx: int):
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame = c.read()
                    if ret and frame is not None:
                        return c
                    c.release()
                    c = cv2.VideoCapture(idx, backend)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = c.read()
                        if ret and frame is not None:
                            return c
                    c.release()
            except Exception:
                pass
        return None

    if str(source).lower() == "auto":
        print("[Orchestrator] Probing physical camera hardware for live real-time detection...")
        cap = open_hardware_camera(0)
        if cap is not None:
            source = 0
            source_type = "LIVE_WEBCAM"
            print("[Orchestrator] Active hardware camera #0 detected! Operating in LIVE REAL-TIME DETECTION mode.")

        if cap is None:
            print("[Orchestrator] Notice: No physical camera accessible. Falling back to recorded demo clip...")
            fallback_source = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
            if not os.path.exists(fallback_source):
                print(f"[Orchestrator] Generating simulation video '{fallback_source}'...")
                from tools.generate_synthetic_data import generate_experiment_video
                generate_experiment_video(fallback_source, anomaly=False)
            cap = cv2.VideoCapture(fallback_source)
            source = fallback_source
            source_type = "RECORDED_CLIP"
    elif str(source).isdigit():
        cam_idx = int(source)
        cap = open_hardware_camera(cam_idx)
        if cap is not None:
            source = cam_idx
            source_type = "LIVE_WEBCAM"
        elif cam_idx != 0:
            print(f"[Warning] Camera index #{cam_idx} not responding. Probing Camera #0...")
            cap = open_hardware_camera(0)
            if cap is not None:
                source = 0
                source_type = "LIVE_WEBCAM"
                print("[Orchestrator] Successfully engaged active Camera #0!")

        if cap is None or not cap.isOpened():
            print(f"[Warning] Web Camera device #{source} could not be opened (disconnected or in use).")
            fallback_source = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
            if not os.path.exists(fallback_source):
                from tools.generate_synthetic_data import generate_experiment_video
                generate_experiment_video(fallback_source, anomaly=False)
            print(f"[Orchestrator] Automatically falling back to video: '{fallback_source}'...")
            cap = cv2.VideoCapture(fallback_source)
            source = fallback_source
            source_type = "RECORDED_CLIP"
        else:
            cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Orchestrator] Ingesting from Web Camera device #{source} ({cw}x{ch}, buffer=1 real-time)...")
    else:
        if not os.path.exists(source):
            print(f"[Orchestrator] Video file '{source}' not found. Generating default simulation...")
            from tools.generate_synthetic_data import generate_experiment_video
            generate_experiment_video(source, anomaly=False)
        cap = cv2.VideoCapture(source)
        source_type = "RED_YELLOW" if "red_yellow" in str(source).lower() else "RECORDED_CLIP"
        print(f"[Orchestrator] Ingesting from Video File: {source}...")

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        agent_monitoring.close()
        return

    # If active camera, configure real-time zero-delay buffer
    if source_type == "LIVE_WEBCAM":
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    frame_id = 0
    t_start = time.time()
    prev_frame_time = t_start
    har_records = [] if record_har_dataset else None

    print("[Orchestrator] Multi-Agent Pipeline Running. Press Ctrl+C or close window to exit.")
    if enable_streaming:
        print(f"[*] Open Browser Mission Dashboard at: http://localhost:{stream_port}/")

    try:
        while True:
            # Check manual source switch request from Web Dashboard
            new_src_req = agent_monitoring.check_source_switch_requested()
            if new_src_req:
                print(f"\n[Orchestrator] Video source switch requested from UI: {new_src_req}")
                try:
                    if new_src_req in ("0", "cam", "webcam"):
                        new_cap = open_hardware_camera(0)
                        if new_cap is not None:
                            if cap is not None:
                                cap.release()
                            cap = new_cap
                            source = 0
                            source_type = "LIVE_WEBCAM"
                            print("[Orchestrator] Switched active video source to LIVE WEBCAM #0.")
                        else:
                            print("[Orchestrator] Hardware camera #0 not available for switch.")
                    elif new_src_req in ("clip1.mp4", "clip", "demo"):
                        target_clip = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
                        new_cap = cv2.VideoCapture(target_clip)
                        if new_cap.isOpened():
                            if cap is not None:
                                cap.release()
                            cap = new_cap
                            source = target_clip
                            source_type = "RECORDED_CLIP"
                            print(f"[Orchestrator] Switched active video source to RECORDED DEMO ({target_clip}).")
                    elif os.path.exists(new_src_req):
                        target_clip = new_src_req
                        new_cap = cv2.VideoCapture(target_clip)
                        if new_cap.isOpened():
                            if cap is not None:
                                cap.release()
                            cap = new_cap
                            source = target_clip
                            source_type = "RED_YELLOW" if "red_yellow" in target_clip.lower() else "RECORDED_CLIP"
                            print(f"[Orchestrator] Switched active video source to RECORDED CLIP ({target_clip}).")
                    agent_validation.reset()
                    agent_har.reset()
                    agent_monitoring.reset()
                    agent_perception.last_lid_angle = 0.0
                    frame_id = 0
                    prev_frame_time = time.time()
                except Exception as e:
                    print(f"[Orchestrator] Source switch error: {e}")

            # Check manual experiment switch request from Web Dashboard
            new_exp_req = agent_monitoring.check_experiment_switch_requested()
            if new_exp_req:
                print(f"\n[Orchestrator] Experiment switch requested from UI: {new_exp_req}")
                try:
                    agent_validation = ValidationAgent(config_path=new_exp_req)
                    agent_validation.load_protocol(new_exp_req)
                    agent_reasoning = ReasoningAgent(config_path=new_exp_req)
                    if ("red_yellow" in new_exp_req or "experiment_fsm" in new_exp_req):
                        is_red_yellow = True
                        agent_monitoring.switch_output_dir(output_dir="experiments_red_yellow", realtime_feed_dir="realtime_feed_red_yellow")
                        if os.path.exists("red_yellow.mp4") and source_type in ("RECORDED_CLIP", "RED_YELLOW"):
                            target_clip = "red_yellow.mp4"
                            new_cap = cv2.VideoCapture(target_clip)
                            if new_cap.isOpened():
                                if cap is not None:
                                    cap.release()
                                cap = new_cap
                                source = target_clip
                                source_type = "RED_YELLOW"
                                print(f"[Orchestrator] Auto-switched video feed to {target_clip} for Red-Yellow procedure.")
                    elif "box_return_fsm" in new_exp_req:
                        is_red_yellow = False
                        agent_monitoring.switch_output_dir(output_dir="experiments", realtime_feed_dir="realtime_feed")
                        if source_type in ("RECORDED_CLIP", "RED_YELLOW"):
                            target_clip = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
                            if os.path.exists(target_clip):
                                new_cap = cv2.VideoCapture(target_clip)
                                if new_cap.isOpened():
                                    if cap is not None:
                                        cap.release()
                                    cap = new_cap
                                    source = target_clip
                                    source_type = "RECORDED_CLIP"
                                    print(f"[Orchestrator] Auto-switched video feed to {target_clip} for Box Return procedure.")
                    if "red_yellow" in new_exp_req:
                        llm_verifier.set_protocol("BAS-EXP-RED-YELLOW")
                    elif "box_return" in new_exp_req:
                        llm_verifier.set_protocol("BAS-EXP-BOX-RETURN")
                    reset_pipeline()
                    print(f"[Orchestrator] Active procedure switched to {agent_validation.experiment_id} ({new_exp_req}).")
                except Exception as e:
                    print(f"[Orchestrator] Experiment switch error: {e}")

            # Check manual reset request from web client or desktop GUI
            web_reset = enable_streaming and agent_monitoring.check_reset_requested()
            gui_reset = desktop_gui and desktop_gui.check_reset_requested()
            if web_reset or gui_reset:
                print("\n[Orchestrator] Reset requested from UI. Restarting real-time test from Step 0...")
                if not str(source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_pipeline()
                frame_id = 0
                prev_frame_time = time.time()

            ret, raw_frame = cap.read()
            if not ret:
                # On live webcam, momentarily dropped frames should not exit or loop
                if str(source).isdigit() or source_type == "LIVE_WEBCAM":
                    time.sleep(0.01)
                    continue

                # If experiment reached COMPLETE, hold final completed state for 3s so user/web client sees completion
                if agent_validation.current_step in (FSMStep.COMPLETE, FSMStep.BOX_CLOSED) or int(agent_validation.current_step) >= (5 if is_red_yellow else 4):
                    print("\n[Orchestrator] Procedure COMPLETED successfully! Holding final state for 3 seconds...")
                    t_hold_end = time.time() + 3.0
                    while time.time() < t_hold_end:
                        if show_window and not desktop_gui:
                            k = cv2.waitKey(20) & 0xFF
                            if k == ord('q'):
                                break
                        if desktop_gui and desktop_gui.is_alive():
                            try:
                                if desktop_gui.root is not None:
                                    desktop_gui.root.update_idletasks()
                                    desktop_gui.root.update()
                            except Exception:
                                pass
                        if agent_monitoring.check_reset_requested():
                            break
                        time.sleep(0.02)

                # Prevent looping to gracefully close the video file and trigger mesh recovery
                print("\n[Orchestrator] Reached end of video file. Gracefully exiting to finalize processing...")
                break

            frame_id += 1
            if max_frames and frame_id > max_frames:
                break

            now = time.time()
            dt = max(1e-4, now - prev_frame_time)
            fps = 1.0 / dt
            prev_frame_time = now

            # ==========================================
            # PIPELINE EXECUTION ACROSS THE 8 AGENTS
            # ==========================================
            t_infer_start = time.time()

            # AGENT 0.5: RelateAnything Spatial Relations
            spatial_relations = []
            if relate_pipe is not None:
                try:
                    res = relate_pipe(raw_frame)
                    for t in res.triplets:
                        sub = t.subject_label.lower()
                        obj = t.object_label.lower()
                        if "box" in sub or "container" in sub or "box" in obj or "container" in obj:
                            spatial_relations.append(f"[{sub}] is {t.predicate} [{obj}] (score: {t.score:.2f})")
                except Exception as e:
                    print(f"\n[DEBUG RelateAnything error]: {e}")
                    pass

            # AGENT 1: Perception Agent (YOLOv8n + 3D HMR)
            objects_cam, pose_cam, lid_angle = agent_perception.process_frame(raw_frame)

            # AGENT 2: IMU Agent (128 Hz ingestion / virtual kinematics + ZUPT)
            imu_telemetry = agent_imu.update_from_vision(pose_cam)

            # AGENT 3: Fusion Agent (Constrained UKF + Stationary Rack Frame R + ROM)
            fused_pose, objects_rack = agent_fusion.fuse_kinematics(
                pose_cam, imu_telemetry, objects_cam, dt=dt
            )

            # AGENT 4: HAR Agent (AdaSpot RoI + 3D Hand-Object Interaction Engine)
            spatial_metrics = agent_spatial.evaluate_spatial_metrics(fused_pose, objects_rack)
            active_hoi, objects_state, current_activity = agent_har.evaluate_interactions(
                fused_pose, objects_rack, lid_angle, spatial_metrics
            )
            # Push Telemetry Snapshot to Real-Time Local LLM Verifier
            comp_obj = objects_state.get("component_box")
            is_inside = comp_obj.is_inside_container if comp_obj else True
            w_joint = fused_pose.joints.get("wrist")
            h_dist = w_joint.pos_rack.distance_to(comp_obj.pos_rack) if (w_joint and comp_obj) else 0.50
            primary_hoi = active_hoi[0].action.value if active_hoi else "IDLE"

            is_priority = (agent_validation.anomaly_status != AnomalyType.NONE) or (agent_validation.debounce_counter > 0)
            detected_names = list(objects_state.keys())

            llm_verifier.push_telemetry(
                frame_id=frame_id,
                step=agent_validation.current_step,
                activity=current_activity,
                lid_angle=lid_angle,
                is_inside=is_inside,
                hoi_action=primary_hoi,
                hand_dist_m=h_dist,
                anomaly=agent_validation.anomaly_status,
                frame=raw_frame,
                detected_objects=detected_names,
                spatial_relations=spatial_relations,
                force_priority=is_priority
            )
            llm_verif = llm_verifier.get_latest_verification()

            # AGENT 5: Digital Twin Agent (3D Scene Synchronization & Render)
            scene_graph = agent_twin.sync_scene_state(
                fused_pose, objects_state, lid_angle, agent_validation.current_step
            )
            twin_canvas = agent_twin.render_digital_twin_canvas(scene_graph)

            # AGENT 6: Validation Agent (Deterministic FSM + Adaptive Debounce + Local LLM Consensus)
            step, deb_count, anomaly, anomaly_msg, trans_event = agent_validation.evaluate_step(
                objects_state, lid_angle, active_hoi, frame_id, llm_verification=llm_verif,
                action_history=agent_har.action_history, astronaut_pose=fused_pose
            )

            # AGENT 7: Reasoning & Guidance Agent (Next-step suggestions + Anomaly alerts)
            instruction, voice_alert = agent_reasoning.evaluate_guidance(
                current_step=step,
                anomaly=anomaly,
                transition_event=trans_event,
                vlm_explanation=agent_validation.vlm_anomaly_explanation,
                is_step_correct=agent_validation.is_step_correct,
                step_verdict=agent_validation.step_verdict,
                anomaly_message=anomaly_msg
            )

            # Record HAR Dataset Snapshot if enabled
            if har_records is not None:
                j_arr = np.zeros((17, 3), dtype=np.float32)
                if fused_pose.keypoints_2d:
                    c_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                               "left_wrist", "right_wrist", "left_hip", "right_hip",
                               "left_knee", "right_knee", "left_ankle", "right_ankle"]
                    for ji, jn in enumerate(c_names):
                        if jn in fused_pose.keypoints_2d:
                            px, py, _ = fused_pose.keypoints_2d[jn]
                            j_arr[ji] = [(px - agent_perception.cx) * 1.5 / agent_perception.fx,
                                         (py - agent_perception.cy) * 1.5 / agent_perception.fy,
                                         1.5]
                har_records.append({
                    "frame_id": frame_id,
                    "timestamp_sec": round(time.time() - t_start, 3),
                    "joints_3d": j_arr,
                    "activity": current_activity,
                    "step_id": int(step)
                })

            latency_ms = (time.time() - t_infer_start) * 1000.0

            # Update Central Shared Memory (Digital Twin Memory Blackboard)
            blackboard.update_frame_metadata(frame_id, fps, latency_ms)
            blackboard.update_objects(objects_state, lid_angle)
            blackboard.update_pose(fused_pose)
            blackboard.update_hoi(active_hoi, current_activity)
            blackboard.update_fsm_state(step, deb_count, anomaly, anomaly_msg, instruction)
            blackboard.update_llm_verification(llm_verif)

            # AGENT 8: Monitoring Agent (Dual Video + Offline TTS + JSONL + HUD Overlay + Action Logger)
            annotated_frame = agent_monitoring.process_egress(
                raw_frame=raw_frame,
                fused_pose=fused_pose,
                objects=objects_state,
                lid_angle=lid_angle,
                current_step=step,
                debounce_count=deb_count,
                anomaly=anomaly,
                instruction=instruction,
                voice_alert=voice_alert,
                transition_event=trans_event,
                frame_id=frame_id,
                fps=fps,
                latency_ms=latency_ms,
                twin_canvas=twin_canvas,
                current_activity=current_activity,
                llm_verification=llm_verif,
                source_type=source_type,
                is_step_correct=agent_validation.is_step_correct,
                step_verdict=agent_validation.step_verdict,
                experiment_id=agent_validation.experiment_id,
                scene_graph=scene_graph,
                spatial_relations=spatial_relations
            )

            # On-Screen Video Feed Display (Native OpenCV Window, only if Desktop GUI is NOT active)
            if show_window and not desktop_gui:
                try:
                    cv2.imshow("BAS Mission Control - Live Stream & Action Detection", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Orchestrator] Exit requested by user (Q key pressed).")
                        break
                    elif key == ord('r'):
                        print("\n[Orchestrator] Manual reset requested (R key pressed).")
                        if not str(source).isdigit():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        reset_pipeline()
                        frame_id = 0
                    elif key == ord('a'):
                        print("\n[Orchestrator] Triggering Offline LLM Procedural Audit (A key pressed)...")
                        from src.llm.offline_llm_analyzer import analyze_session
                        analyze_session()
                except Exception:
                    pass

            # Update Desktop Tkinter GUI if active
            if desktop_gui:
                if not desktop_gui.is_alive():
                    print("\n[Orchestrator] Desktop GUI window closed by user.")
                    break
                desktop_gui.update_frame(annotated_frame)
                log_snippet = f"[{trans_event}] {instruction}" if trans_event else None
                desktop_gui.update_state(int(step), instruction, anomaly.value, log_snippet)

            # Procedure Step Event Commit
            if trans_event:
                print(f"\n[PROCEDURE EVENT] Milestone Committed: {trans_event} -> Step {int(step)} ({step.get_name(is_red_yellow)})")

            # Procedural Completion: Trigger Automated Offline Local LLM Audit (Async / Non-Blocking)
            if trans_event in ("BOX_CLOSED", "YELLOW_BOX_EXTRACTED", "BENCHMARK_COMPLETE"):
                print("\n" + "=" * 70)
                print(f"[Orchestrator] PROCEDURE COMPLETED ({trans_event})! Launching Non-Blocking AI Mission Audit in background...")
                print("=" * 70)
                try:
                    from src.llm.offline_llm_analyzer import start_async_analysis
                    start_async_analysis()
                except Exception as e:
                    print(f"[Orchestrator] LLM Audit notice: {e}")

            # Clean in-place console status line
            if frame_id % 5 == 0:
                llm_step_int = llm_verif.get("verified_step", int(step))
                llm_conf_pct = int(llm_verif.get("confidence", 0.90) * 100)
                llm_disp = f"S{llm_step_int} ({llm_conf_pct}%)"
                v_disp = "OK" if agent_validation.is_step_correct else "ERR"
                vlm_wrong = llm_verif.get("what_is_wrong", "None")
                wrong_disp = f" | Issue: {vlm_wrong[:35]}" if (vlm_wrong != "None" and not agent_validation.is_step_correct) else ""
                sys.stdout.write(
                    f"\r[{source_type[:4]}] F{frame_id:04d} | Step {int(step)}: {step.get_name(is_red_yellow):16s} | Verdict: {v_disp} | LLM: {llm_disp:10s} | Deb: {deb_count:02d}/06 | FPS: {fps:4.1f}{wrong_disp}  "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutdown requested by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Orchestrator] Pipeline terminated: {e}")
    finally:
        try:
            llm_verifier.close()
        except (Exception, KeyboardInterrupt):
            pass

        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
        except (Exception, KeyboardInterrupt):
            pass

        try:
            cv2.destroyAllWindows()
        except (Exception, KeyboardInterrupt):
            pass

        try:
            agent_monitoring.close()
        except (Exception, KeyboardInterrupt):
            pass

        # Trigger mesh recovery after video file is fully closed/finalized
        try:
            from src.core.mesh_recovery import start_async_mesh_recovery
            video_to_process = None
            if hasattr(agent_monitoring, 'video_pipeline') and agent_monitoring.video_pipeline and agent_monitoring.video_pipeline.local_output_path:
                video_to_process = agent_monitoring.video_pipeline.local_output_path
            elif source_type != "LIVE_WEBCAM" and isinstance(source, str):
                video_to_process = source
            
            if video_to_process:
                print(f"\n[Orchestrator] Automatically triggering Mesh Recovery on finalized video: {video_to_process}")
                start_async_mesh_recovery(video_to_process)
        except Exception as e:
            print(f"[Orchestrator] Mesh Recovery finalization notice: {e}")
        
        # Calculate final telemetry compression audit
        try:
            total_time_sec = max(0.1, time.time() - t_start)
            ratio = agent_monitoring.telemetry.calculate_compression_ratio(total_time_sec)
            print("=" * 70)
            print("MISSION TELEMETRY AUDIT")
            print(f"Total Session Duration : {total_time_sec:.1f} seconds")
            print(f"Total Frames Processed : {frame_id} frames")
            print(f"Structured JSONL Size  : {agent_monitoring.telemetry.total_bytes_written} bytes")
            print(f"Empirical Compression : {ratio:,.1f} : 1")
            print(f"Telemetry Log Saved At : {agent_monitoring.telemetry.output_path}")
            if hasattr(agent_monitoring, "csv_logger") and agent_monitoring.csv_logger:
                print(f"3D/Twin CSV Telemetry : {agent_monitoring.csv_logger.output_path}")
                print(f"Latest 3D Telemetry CSV: {agent_monitoring.csv_logger.latest_symlink_path}")
                print(f"Dedicated Real-Time CSV: {agent_monitoring.csv_logger.realtime_current_path}")
            if record_har_dataset and har_records:
                npz_out = os.path.join(realtime_feed_dir, "har_session_joints.npz")
                np.savez_compressed(
                    npz_out,
                    joints_3d=np.array([r["joints_3d"] for r in har_records], dtype=np.float32),
                    activities=np.array([r["activity"] for r in har_records]),
                    step_ids=np.array([r["step_id"] for r in har_records], dtype=np.int32),
                    timestamps=np.array([r["timestamp_sec"] for r in har_records], dtype=np.float32)
                )
                print(f"Session HAR Dataset NPZ : {npz_out}")
            print("=" * 70)
        except (Exception, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAS Multi-Agent HAR Orchestrator")
    parser.add_argument("--source", default="auto",
                        help="Video source: 'auto' (default: probes physical webcam #0, falls back to demo clip), camera index (0), or path to MP4")
    parser.add_argument("--config", default=None,
                        help="Path to FSM configuration file (default: auto-detected)")
    parser.add_argument("--desktop-gui", action="store_true",
                        help="Launch native desktop Tkinter GUI")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable native OpenCV desktop display window")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable voice synthesis")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable IP streaming server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for Web Mission Dashboard & RTSP/HTTP stream")
    parser.add_argument("--frames", type=int, default=None,
                        help="Maximum frames to process (useful for automated testing)")
    parser.add_argument("--realtime-dir", type=str, default="realtime_feed",
                        help="Dedicated folder for real-time video feed CSV telemetry (default: realtime_feed)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dedicated folder for experiment CSV telemetry (default: auto-detected, 'experiments' or 'experiments_red_yellow')")
    parser.add_argument("--common", action="store_true",
                        help="Launch real-time Common Object Detection mode (COCO-80 classes: phone, bottle, cup, person, book, etc.)")
    parser.add_argument("--record-har-dataset", action="store_true",
                        help="Record and export synchronized 3D skeleton and HAR action dataset (.npz)")
    args = parser.parse_args()

    if args.common:
        from tools.realtime_detect import run_realtime_detection
        run_realtime_detection(source=args.source, conf_threshold=0.30, max_frames=args.frames)
        sys.exit(0)

    try:
        # If desktop GUI is requested, suppress the duplicate raw OpenCV window
        show_win = not args.no_window and not args.desktop_gui
        run_orchestrator(
            source=args.source,
            config_path=args.config,
            use_desktop_gui=args.desktop_gui,
            show_window=show_win,
            enable_tts=not args.no_tts,
            enable_streaming=not args.no_stream,
            stream_port=args.port,
            max_frames=args.frames,
            realtime_feed_dir=args.realtime_dir,
            output_csv_dir=args.output_dir,
            record_har_dataset=args.record_har_dataset
        )
    except KeyboardInterrupt:
        sys.exit(0)

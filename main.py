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
_venv_candidates = [
    os.path.join(_workspace_dir, ".venv", "bin", "python"),
    os.path.join(_workspace_dir, ".venv", "Scripts", "python.exe"),
]
_venv_python = next((p for p in _venv_candidates if os.path.exists(p)), None)
if _venv_python and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([_venv_python] + sys.argv))

import glob
_venv_site_candidates = [
    os.path.join(_workspace_dir, ".venv", "Lib", "site-packages"),
    *glob.glob(os.path.join(_workspace_dir, ".venv", "lib", "python*", "site-packages")),
]
for _site in _venv_site_candidates:
    if os.path.exists(_site) and _site not in sys.path:
        sys.path.insert(0, _site)

import math
import time
import argparse
import cv2
import numpy as np


def is_display_available() -> bool:
    """
    Checks if a graphical display environment (X11, Wayland, Windows, macOS) is available.
    Returns False on headless Linux environments where DISPLAY or WAYLAND_DISPLAY is not set.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# RelateAnything is now integrated directly
try:
    from deploy.postprocess import ThresholdConfig # type: ignore
    from deploy.runtime import DetectorConfig, ScenePipeline # type: ignore
except ImportError as e:
    print(f"CRITICAL: Failed to import ScenePipeline: {e}")
    ScenePipeline = None

# Core & Blackboard
from src.core.types import FSMStep, AnomalyType, EntityState
from src.core.shared_memory import DigitalTwinBlackboard

# 8 Specialized Agents + Real-Time Local LLM Verifier
from src.agents.perception_agent import PerceptionAgent
from src.agents.scene_object_agent import SceneObjectAgent
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
    record_har_dataset=False,
    device=None,
    enable_scene_objects=True,
    vlm_model=None
):
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - ON-BOARD HAR MULTI-AGENT SYSTEM")
    print("   Autonomous Sequence Validation & Digital Twin Architecture")
    print("   ISRO SIH Problem Statement #26174")
    print("=" * 70)

    # Check display availability for native GUI windows
    display_ok = is_display_available()
    if show_window and not display_ok:
        print("[Orchestrator] Notice: No graphical display detected (headless environment). Native desktop window disabled.")
        print(f"[*] Access the real-time stream and mission dashboard at: http://localhost:{stream_port}/")
        show_window = False

    if use_desktop_gui and not display_ok:
        print("[Orchestrator] Notice: No graphical display detected. Disabling desktop Tkinter GUI.")
        use_desktop_gui = False

    # Detect if source or protocol is the Water-Bottle or Red-Yellow experiment
    is_water_bottle = (
        "bottle" in str(source).lower()
        or (config_path is not None and "bottle" in str(config_path).lower())
    )
    is_red_yellow = (
        not is_water_bottle and (
            "red_yellow" in str(source).lower()
            or "clip.mp4" in str(source).lower()
            or (config_path is not None and "red_yellow" in str(config_path).lower())
        )
    )

    if is_water_bottle:
        if config_path is None or "box_return" in str(config_path):
            config_path = "configs/water_bottle_fsm.json"
        if realtime_feed_dir == "realtime_feed":
            realtime_feed_dir = "realtime_feed_bottle"
        if output_csv_dir is None:
            output_csv_dir = "experiments_water_bottle"
    elif is_red_yellow:
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

    # 2. Instantiate Real-Time Asynchronous Local VLM Verifier (Ollama Qwen3-VL)
    vlm_model = vlm_model or os.environ.get("BAS_VLM_MODEL", "qwen3-vl:2b-instruct")
    print(f"[Orchestrator] Engaging Multimodal VLM Real-Time Verifier ({vlm_model} @ http://localhost:11434)...")
    exp_id = "BAS-EXP-WATER-BOTTLE" if is_water_bottle else ("BAS-EXP-RED-YELLOW" if is_red_yellow else "BAS-EXP-BOX-RETURN")
    llm_verifier = RealtimeLLMVerifier(
        model_name=vlm_model,
        experiment_id=exp_id
    )

    # 3. Instantiate the 8 Specialized Agents
    print("[Orchestrator] Initializing 8 Specialized Agents...")
    chosen_detector = "models/water_bottle_yolo26.pt" if (is_water_bottle and os.path.exists("models/water_bottle_yolo26.pt")) else "models/detector_offline.pt"
    agent_perception = PerceptionAgent(model_path=chosen_detector, device=device)
    agent_scene = SceneObjectAgent(device=device) if enable_scene_objects else None
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
        if agent_scene:
            agent_scene.reset()
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
            show_window = is_display_available()

    # Spatial relations log file
    spatial_log_path = os.path.join(output_csv_dir, f"spatial_relations_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    print(f"[Orchestrator] Spatial Relations Log: {spatial_log_path}")
    with open(spatial_log_path, "w") as sf:
        sf.write("Frame ID | Spatial Relations\n")
        sf.write("-" * 50 + "\n")

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

    if str(source).lower() == "browser":
        from src.streaming.browser_capture import BrowserCapture
        browser_port = stream_port + 1
        cap = BrowserCapture(port=browser_port)
        source_type = "LIVE_WEBCAM"
        print(f"[Orchestrator] Browser camera ingest ready. Open http://localhost:{browser_port}/ on the camera device and press 'Start camera'.")
    elif str(source).lower() == "auto":
        print("[Orchestrator] Probing physical camera hardware for live real-time detection...")
        cap = open_hardware_camera(0)
        if cap is not None:
            source = 0
            source_type = "LIVE_WEBCAM"
            print("[Orchestrator] Active hardware camera #0 detected! Operating in LIVE REAL-TIME DETECTION mode.")

        if cap is None:
            print("[Orchestrator] Notice: No physical camera accessible. Falling back to recorded demo clip...")
            if is_water_bottle:
                fallback_source = "demo_bottle.mp4"
                if not os.path.exists(fallback_source):
                    print(f"[Orchestrator] Generating bottle simulation video '{fallback_source}'...")
                    from tools.generate_bottle_synthetic_data import generate_bottle_simulation_video
                    generate_bottle_simulation_video(fallback_source)
            else:
                fallback_source = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
                if not os.path.exists(fallback_source):
                    print(f"[Orchestrator] Generating simulation video '{fallback_source}'...")
                    from tools.generate_synthetic_data import generate_experiment_video
                    generate_experiment_video(fallback_source)
            cap = cv2.VideoCapture(fallback_source)
            source = fallback_source
            source_type = "WATER_BOTTLE" if is_water_bottle else "RECORDED_CLIP"
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
            if is_water_bottle:
                fallback_source = "demo_bottle.mp4"
                if not os.path.exists(fallback_source):
                    from tools.generate_bottle_synthetic_data import generate_bottle_simulation_video
                    generate_bottle_simulation_video(fallback_source)
            else:
                fallback_source = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
                if not os.path.exists(fallback_source):
                    from tools.generate_synthetic_data import generate_experiment_video
                    generate_experiment_video(fallback_source)
            print(f"[Orchestrator] Automatically falling back to video: '{fallback_source}'...")
            cap = cv2.VideoCapture(fallback_source)
            source = fallback_source
            source_type = "WATER_BOTTLE" if is_water_bottle else "RECORDED_CLIP"
        else:
            cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Orchestrator] Ingesting from Web Camera device #{source} ({cw}x{ch}, buffer=1 real-time)...")
    else:
        if not os.path.exists(source):
            if is_water_bottle or "bottle" in str(source).lower():
                print(f"[Orchestrator] Video file '{source}' not found. Generating bottle simulation...")
                from tools.generate_bottle_synthetic_data import generate_bottle_simulation_video
                generate_bottle_simulation_video(source)
            else:
                print(f"[Orchestrator] Video file '{source}' not found. Generating default simulation...")
                from tools.generate_synthetic_data import generate_experiment_video
                generate_experiment_video(source)
        cap = cv2.VideoCapture(source)
        if "bottle" in str(source).lower() or is_water_bottle:
            source_type = "WATER_BOTTLE"
        elif "red_yellow" in str(source).lower():
            source_type = "RED_YELLOW"
        else:
            source_type = "RECORDED_CLIP"
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
                    if ("bottle" in new_exp_req or "water_bottle" in new_exp_req):
                        is_water_bottle = True
                        is_red_yellow = False
                        agent_monitoring.switch_output_dir(output_dir="experiments_water_bottle", realtime_feed_dir="realtime_feed_bottle")
                        if os.path.exists("demo_bottle.mp4") and source_type in ("RECORDED_CLIP", "RED_YELLOW", "WATER_BOTTLE"):
                            target_clip = "demo_bottle.mp4"
                            new_cap = cv2.VideoCapture(target_clip)
                            if new_cap.isOpened():
                                if cap is not None:
                                    cap.release()
                                cap = new_cap
                                source = target_clip
                                source_type = "WATER_BOTTLE"
                                print(f"[Orchestrator] Auto-switched video feed to {target_clip} for Water Bottle procedure.")
                        if os.path.exists("models/water_bottle_yolo26.pt"):
                            agent_perception.load_detector("models/water_bottle_yolo26.pt")
                    elif ("red_yellow" in new_exp_req or "experiment_fsm" in new_exp_req):
                        is_water_bottle = False
                        is_red_yellow = True
                        agent_monitoring.switch_output_dir(output_dir="experiments_red_yellow", realtime_feed_dir="realtime_feed_red_yellow")
                        if agent_perception.is_bottle_model:
                            agent_perception.load_detector("models/detector_offline.pt")
                        if os.path.exists("red_yellow.mp4") and source_type in ("RECORDED_CLIP", "RED_YELLOW", "WATER_BOTTLE"):
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
                        is_water_bottle = False
                        is_red_yellow = False
                        agent_monitoring.switch_output_dir(output_dir="experiments", realtime_feed_dir="realtime_feed")
                        if agent_perception.is_bottle_model:
                            agent_perception.load_detector("models/detector_offline.pt")
                        if source_type in ("RECORDED_CLIP", "RED_YELLOW", "WATER_BOTTLE"):
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
                    if "bottle" in new_exp_req:
                        llm_verifier.set_protocol("BAS-EXP-WATER-BOTTLE")
                    elif "red_yellow" in new_exp_req:
                        llm_verifier.set_protocol("BAS-EXP-RED-YELLOW")
                    elif "box_return" in new_exp_req:
                        llm_verifier.set_protocol("BAS-EXP-BOX-RETURN")
                    reset_pipeline()
                    print(f"[Orchestrator] Active procedure switched to {agent_validation.experiment_id} ({new_exp_req}).")
                except Exception as e:
                    print(f"[Orchestrator] Experiment switch error: {e}")

            # Check manual reset or start request from web client or desktop GUI
            web_reset = enable_streaming and agent_monitoring.check_reset_requested()
            web_start = enable_streaming and hasattr(agent_monitoring, 'check_start_requested') and agent_monitoring.check_start_requested()
            gui_reset = desktop_gui and desktop_gui.check_reset_requested()
            if web_reset or gui_reset:
                print("\n[Orchestrator] Reset requested from UI. Restarting real-time test from Step 0...")
                if not str(source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_pipeline()
                frame_id = 0
                prev_frame_time = time.time()
            elif web_start:
                print("\n[Orchestrator] Start/Restart requested from UI...")
                max_step_thresh = 5 if (is_red_yellow or is_water_bottle) else 4
                if agent_validation.current_step in (FSMStep.COMPLETE, FSMStep.BOX_CLOSED) or int(agent_validation.current_step) >= max_step_thresh:
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

                if enable_streaming:
                    # Procedure reached end of video stream. Hold server alive in completed state so user can Reset or Start from Web UI
                    print("\n[Orchestrator] Procedure COMPLETED or video stream ended. Holding stream server (awaiting Reset or Start from Web UI)...")
                    while True:
                        if agent_monitoring.check_reset_requested():
                            print("\n[Orchestrator] Reset requested from UI. Restarting test sequence from Step 0...")
                            if not str(source).isdigit():
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            reset_pipeline()
                            frame_id = 0
                            prev_frame_time = time.time()
                            break
                        if hasattr(agent_monitoring, 'check_start_requested') and agent_monitoring.check_start_requested():
                            print("\n[Orchestrator] Start requested from UI. Restarting test sequence from Step 0...")
                            if not str(source).isdigit():
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            reset_pipeline()
                            frame_id = 0
                            prev_frame_time = time.time()
                            break
                        if hasattr(agent_monitoring, 'check_source_switch_requested'):
                            if agent_monitoring.video_pipeline and getattr(agent_monitoring.video_pipeline._server, 'source_switch_requested', None):
                                break
                        if show_window and not desktop_gui:
                            k = cv2.waitKey(40) & 0xFF
                            if k == ord('q'):
                                break
                        time.sleep(0.05)
                    continue

                # If experiment reached COMPLETE, hold final completed state for 3s so user/web client sees completion
                if agent_validation.current_step in (FSMStep.COMPLETE, FSMStep.BOX_CLOSED) or int(agent_validation.current_step) >= (5 if (is_red_yellow or is_water_bottle) else 4):
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

            # AGENT 1: Perception Agent (YOLOv8n + 3D HMR)
            objects_cam, pose_cam, lid_angle = agent_perception.process_frame(raw_frame)

            # Extract Spatial Relations from objects (populated by PerceptionAgent using RelateAnything)
            spatial_relations = []
            for obj in objects_cam.values():
                if getattr(obj, 'relations', None):
                    for r in obj.relations:
                        spatial_relations.append(f"[{r.subject_name}] is {r.predicate} [{r.object_name}] (score: {r.score:.2f})")

            # Heuristic spatial relations fallback relative to container
            if not spatial_relations:
                cont = objects_cam.get("container_box")
                if cont and cont.bbox:
                    for obj_name, obj in objects_cam.items():
                        if obj_name in ["container_box", "container_lid", "hand", "operator_hand", "human_body", "person", "astronaut"]:
                            continue
                        if obj.bbox:
                            cx, cy = obj.bbox.centroid
                            if cont.bbox.xmin <= cx <= cont.bbox.xmax and cont.bbox.ymin <= cy <= cont.bbox.ymax:
                                spatial_relations.append(f"[{obj_name}] is inside [container_box]")
                            elif cy < cont.bbox.ymin:
                                spatial_relations.append(f"[{obj_name}] is above [container_box]")
                            elif cy > cont.bbox.ymax:
                                spatial_relations.append(f"[{obj_name}] is below [container_box]")
                            elif cx < cont.bbox.xmin:
                                spatial_relations.append(f"[{obj_name}] is left of [container_box]")
                            elif cx > cont.bbox.xmax:
                                spatial_relations.append(f"[{obj_name}] is right of [container_box]")
            
            # Explicit Human Body Spatial Position (Deterministic 2D Depth Heuristic)
            cont = objects_cam.get("container_box")
            human = objects_cam.get("human_body") or objects_cam.get("person") or objects_cam.get("astronaut")
            if cont and cont.bbox and human and human.bbox:
                # Use ymax (feet position) to determine depth occlusion
                # If human feet are higher up in the 2D image than the container bottom, they are behind
                if human.bbox.ymax < cont.bbox.ymax - 20:
                    spatial_relations.insert(0, "[human_body] is behind [container_box]")
                elif human.bbox.ymax > cont.bbox.ymax + 20:
                    spatial_relations.insert(0, "[human_body] is in front of [container_box]")
                else:
                    spatial_relations.insert(0, "[human_body] is beside [container_box]")
            
            # Log spatial relations to separate file
            if spatial_relations:
                with open(spatial_log_path, "a") as sf:
                    sf.write(f"F{frame_id:04d} | {', '.join(spatial_relations)}\n")

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
            # Distance Calculations (Hand to Boxes)
            comp_obj = objects_state.get("component_box")
            is_inside = comp_obj.is_inside_container if comp_obj else True
            w_joint = fused_pose.joints.get("right_wrist") or fused_pose.joints.get("wrist")
            h_dist = w_joint.pos_rack.distance_to(comp_obj.pos_rack) if (w_joint and comp_obj) else 0.50
            
            red_box = objects_state.get("red_box")
            yellow_box = objects_state.get("yellow_box")
            dist_red = w_joint.pos_rack.distance_to(red_box.pos_rack) if (w_joint and red_box) else -1.0
            dist_yellow = w_joint.pos_rack.distance_to(yellow_box.pos_rack) if (w_joint and yellow_box) else -1.0
            
            # Log exact 3D distances to CSV (Real-Time)
            dist_csv_path = os.path.join(output_csv_dir, "hand_box_distances_realtime.csv")
            if not os.path.exists(dist_csv_path):
                with open(dist_csv_path, "w") as df:
                    df.write("frame,timestamp_sec,hand_red_box_dist_m,hand_yellow_box_dist_m\n")
            with open(dist_csv_path, "a") as df:
                df.write(f"{frame_id},{round(time.time() - t_start, 3)},{dist_red:.3f},{dist_yellow:.3f}\n")

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

            # Synchronize real-time activity from VLM Verifier and semantic relations
            procedural_stems = (
                "IDLE", "OBSERVING", "CONTAINER", "BOX", "LID", "COMPONENT",
                "RED", "YELLOW", "GRASP", "PICK", "RETURN", "EXTRACT",
                "APPROACH", "CONTACT", "HOLD", "DOCKED", "STANDBY",
                "BOTTLE", "CAP", "DRINK", "DRINKING", "WATER", "POUR", "TAKE", "PUT"
            )

            if llm_verif:
                vlm_act = llm_verif.get("detected_activity")
                vlm_verdict = llm_verif.get("anomaly_verdict", "NOMINAL")
                what_wrong = str(llm_verif.get("what_is_wrong", "")).lower()

                if vlm_act and vlm_act.upper() not in ("IDLE", "OBSERVING", "NONE", ""):
                    is_vlm_proc = any(stem in vlm_act.upper() for stem in procedural_stems)
                    if vlm_verdict in ("PROCEDURAL_ERROR", "PHYSICAL_ANOMALY") or not is_vlm_proc:
                        current_activity = vlm_act.upper()
                elif vlm_verdict in ("PROCEDURAL_ERROR", "PHYSICAL_ANOMALY"):
                    for kw, label in [
                        ("danc", "DANCING"), ("walk", "WALKING AWAY"), ("pacing", "PACING / WALKING"),
                        ("wav", "WAVING"), ("clap", "CLAPPING"), ("stretch", "STRETCHING"),
                        ("phone", "USING PHONE"), ("drink", "DRINKING"), ("eat", "EATING"),
                        ("jump", "JUMPING"), ("point", "POINTING / GESTURING"), ("gestur", "GESTURING"),
                        ("scratch", "TOUCHING FACE / HEAD"), ("face", "TOUCHING FACE / HEAD"),
                        ("head", "TOUCHING FACE / HEAD"), ("cross", "ARMS CROSSED"),
                        ("hip", "HANDS ON HIPS"), ("rais", "HANDS RAISED"), ("talk", "TALKING"),
                        ("watch", "LOOKING AT WATCH"), ("bend", "BENDING DOWN"), ("exercis", "EXERCISING")
                    ]:
                        if kw in what_wrong:
                            current_activity = label
                            break
                    else:
                        if "is " in what_wrong:
                            extracted = what_wrong.split("is ", 1)[1].split(".")[0].split(",")[0].strip().upper()
                            if extracted:
                                current_activity = extracted

            # Check semantic relations for anomalous non-procedural activity
            for obj in objects_state.values():
                if hasattr(obj, "relations") and obj.relations:
                    for r in obj.relations:
                        pred = r.predicate.upper()
                        if pred not in ("HOLDING", "TOUCHING", "LOOKING AT", "STANDING NEXT TO", "NEAR", "NONE"):
                            current_activity = pred

            # Kinematic & Spatial Evaluation for Water Bottle Manipulation Experiment
            if is_water_bottle:
                b_obj = objects_state.get("bottle")
                c_obj = objects_state.get("bottle_cap") or objects_state.get("cap")
                hand_obj = objects_state.get("operator_hand") or objects_state.get("hand")
                
                if b_obj and b_obj.bbox:
                    b_cx, b_cy = b_obj.bbox.centroid
                    b_top_x = b_cx
                    b_top_y = b_obj.bbox.ymin
                    b_w = b_obj.bbox.xmax - b_obj.bbox.xmin
                    b_h = b_obj.bbox.ymax - b_obj.bbox.ymin

                    # 1. Surface resting vs held / elevated
                    if hand_obj and hand_obj.bbox:
                        h_cx, h_cy = hand_obj.bbox.centroid
                        d_hb = math.hypot(h_cx - b_cx, h_cy - b_cy)
                        is_hand_near = d_hb < max(120.0, b_w * 2.2)
                    else:
                        is_hand_near = (b_obj.bbox.ymax < 540)

                    if b_obj.bbox.ymax < 540 or (is_hand_near and b_cy < 510):
                        b_obj.on_surface = False
                        b_obj.state = EntityState.EXTRACTED
                    else:
                        b_obj.on_surface = True
                        b_obj.state = EntityState.DOCKED

                    # 2. Cap state (Open vs Closed)
                    if c_obj and c_obj.bbox:
                        c_cx, c_cy = c_obj.bbox.centroid
                        dist_top_cap = math.hypot(c_cx - b_top_x, c_cy - b_top_y)
                        if dist_top_cap > 55.0:
                            b_obj.cap_open = True
                            c_obj.state = EntityState.EXTRACTED
                        else:
                            b_obj.cap_open = False
                            c_obj.state = EntityState.DOCKED
                    elif not b_obj.on_surface and agent_validation.current_step in (FSMStep.CAP_OPENED, FSMStep.DRINKING):
                        b_obj.cap_open = True

                    # 3. Distance to mouth & drinking tilt
                    mouth_pt = None
                    if fused_pose.keypoints_2d and "nose" in fused_pose.keypoints_2d:
                        nx, ny, _ = fused_pose.keypoints_2d["nose"]
                        mouth_pt = (nx, ny + 15)
                    elif objects_state.get("human_body") and objects_state["human_body"].bbox:
                        hb = objects_state["human_body"].bbox
                        mouth_pt = (hb.centroid[0], hb.ymin + 60)

                    if mouth_pt:
                        d_mouth = math.hypot(b_top_x - mouth_pt[0], b_top_y - mouth_pt[1])
                        b_obj.dist_to_mouth = d_mouth / 650.0
                        b_obj.near_mouth = (d_mouth < 130.0)
                    else:
                        b_obj.dist_to_mouth = 0.50
                        b_obj.near_mouth = False

                    # 4. Tilt Angle
                    if b_obj.near_mouth:
                        b_obj.tilt_angle = 35.0
                    else:
                        aspect = b_w / max(1.0, b_h)
                        b_obj.tilt_angle = min(75.0, max(0.0, (aspect - 0.30) * 120.0))

            # AGENT 5: Digital Twin Agent (3D Scene Synchronization & Render)
            scene_graph = agent_twin.sync_scene_state(
                fused_pose, objects_state, lid_angle, agent_validation.current_step
            )
            twin_canvas = agent_twin.render_digital_twin_canvas(scene_graph)

            # AGENT 6: Validation Agent (Deterministic FSM + Adaptive Debounce + Local LLM Consensus)
            step, deb_count, anomaly, anomaly_msg, trans_event = agent_validation.evaluate_step(
                objects_state, lid_angle, active_hoi, frame_id,
                llm_verification=llm_verif,
                action_history=agent_har.action_history,
                astronaut_pose=fused_pose,
                current_activity=current_activity
            )

            # Preserve ongoing procedural step and reflect any detected non-procedural activity across attributes
            is_cur_proc = any(stem in current_activity.upper() for stem in procedural_stems)
            if not is_cur_proc and current_activity.upper() not in ("IDLE", "OBSERVING", "STANDBY", ""):
                clean_act = current_activity.replace("_", " ").strip()
                anomaly = AnomalyType.ERROR_SEQ
                anomaly_msg = f"Warning: Wrong move! {clean_act} is not part of the procedure."
                agent_validation.anomaly_status = AnomalyType.ERROR_SEQ
                agent_validation.anomaly_message = anomaly_msg
                agent_validation.is_step_correct = False
                agent_validation.step_verdict = f"ANOMALY: {clean_act} [NOT IN PROCEDURE]"

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

            # Open-vocabulary scene objects & human-object interactions (display / telemetry only)
            scene_objects = []
            if agent_scene and agent_scene.enabled:
                scene_objects = agent_scene.process(raw_frame, fused_pose, objects_state)
                scene_activity = SceneObjectAgent.summarize(scene_objects)
                if scene_activity and current_activity == "IDLE":
                    current_activity = scene_activity
                SceneObjectAgent.draw(raw_frame, scene_objects)

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
                spatial_relations=spatial_relations,
                scene_objects=scene_objects
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
            else:
                # Pace playback for recorded video in headless mode so web stream is smooth
                if source_type in ("RECORDED_CLIP", "RED_YELLOW"):
                    time.sleep(0.015)

            # Update Desktop Tkinter GUI if active
            if desktop_gui:
                if not desktop_gui.is_alive():
                    print("\n[Orchestrator] Desktop GUI window closed by user.")
                    break
                desktop_gui.update_frame(annotated_frame)
                log_snippet = f"[{trans_event}] {instruction}" if trans_event else None
                desktop_gui.update_state(int(step), instruction, anomaly.value, log_snippet)

            # Procedure Step Event Commit
            step_variant = "bottle" if is_water_bottle else ("dual" if is_red_yellow else False)
            if trans_event:
                print(f"\n[PROCEDURE EVENT] Milestone Committed: {trans_event} -> Step {int(step)} ({step.get_name(step_variant)})")

            # Procedural Completion: Trigger Automated Offline Local LLM Audit (Async / Non-Blocking)
            if trans_event in ("BOX_CLOSED", "YELLOW_BOX_EXTRACTED", "BENCHMARK_COMPLETE", "BOTTLE_PUT_DOWN", "DRINKING_COMPLETE"):
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
                    f"\r[{source_type[:4]}] F{frame_id:04d} | Step {int(step)}: {step.get_name(step_variant):16s} | Verdict: {v_disp} | LLM: {llm_disp:10s} | Deb: {deb_count:02d}/06 | FPS: {fps:4.1f}{wrong_disp}  "
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
            if show_window and is_display_available():
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
                        help="Video source: 'auto' (default: probes physical webcam #0, falls back to demo clip), camera index (0), 'browser' (remote webcam streamed from a browser on --port + 1), or path to MP4")
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
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device: 'cuda', 'cuda:0', 'cpu', or None for auto (CUDA if available)")
    parser.add_argument("--no-scene-objects", action="store_true",
                        help="Disable open-vocabulary detection of all scene objects and human-object interactions")
    parser.add_argument("--vlm-model", type=str, default=None,
                        help="Ollama VLM used for real-time verification (default: $BAS_VLM_MODEL or qwen3-vl:2b-instruct)")
    args = parser.parse_args()

    if args.common:
        from tools.realtime_detect import run_realtime_detection
        run_realtime_detection(
            source=args.source,
            conf_threshold=0.30,
            max_frames=args.frames,
            headless=not is_display_available() or args.no_window,
            device=args.device
        )
        sys.exit(0)

    try:
        # Check if graphical display is available
        has_display = is_display_available()
        # If desktop GUI is requested, suppress the duplicate raw OpenCV window
        show_win = not args.no_window and not args.desktop_gui and has_display
        use_gui = args.desktop_gui and has_display
        if args.desktop_gui and not has_display:
            print("[Warning] --desktop-gui requested but no graphical display is available (headless).")

        run_orchestrator(
            source=args.source,
            config_path=args.config,
            use_desktop_gui=use_gui,
            show_window=show_win,
            enable_tts=not args.no_tts,
            enable_streaming=not args.no_stream,
            stream_port=args.port,
            max_frames=args.frames,
            realtime_feed_dir=args.realtime_dir,
            output_csv_dir=args.output_dir,
            record_har_dataset=args.record_har_dataset,
            device=args.device,
            enable_scene_objects=not args.no_scene_objects,
            vlm_model=args.vlm_model
        )
    except KeyboardInterrupt:
        sys.exit(0)

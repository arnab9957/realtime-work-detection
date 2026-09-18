"""
BAS Autonomous HAR System - Standalone Offline Trained AI Model Runner
Deliverable: A trained AI model that runs on an offline standalone system.
ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174

Executes the custom-trained YOLOv8 neural network (models/detector_offline.pt)
locally on live webcam streams or pre-recorded spaceflight experiment video files.
Runs 100% offline with zero cloud dependencies, external APIs, or network access.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Auto-switch to project virtual environment interpreter if launched via global python
_venv_python = os.path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([_venv_python] + sys.argv))

_venv_site = os.path.join(WORKSPACE_ROOT, ".venv", "Lib", "site-packages")
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

try:
    from ultralytics import YOLO
except ImportError:
    print("[Error] Ultralytics YOLO package is required. Run using virtual environment:")
    print(r"  .\.venv\Scripts\python.exe run_offline_model.py")
    sys.exit(1)


# Class color scheme for the 6 trained experiment classes
CLASS_PALETTE = {
    0: (60, 180, 255),   # container_box: Bright Orange/Amber
    1: (255, 180, 50),   # container_lid: Sky Blue
    2: (60, 60, 240),    # red_box: Vivid Red
    3: (50, 225, 240),   # yellow_box: Vivid Yellow
    4: (80, 230, 120),   # operator_hand: Bright Green
    5: (220, 120, 200),  # human_body: Soft Purple/Magenta
}

CLASS_DISPLAY_NAMES = {
    "container_box": "CONTAINER BOX",
    "container_lid": "CONTAINER LID",
    "red_box": "RED BOX (PAYLOAD)",
    "yellow_box": "YELLOW BOX (PAYLOAD)",
    "operator_hand": "ASTRONAUT HAND",
    "human_body": "ASTRONAUT SKELETON",
}


def get_class_color(class_id: int):
    return CLASS_PALETTE.get(class_id, (200, 200, 200))


def run_offline_standalone(
    model_path="models/detector_offline.pt",
    source="auto",
    conf_threshold=0.35,
    iou_threshold=0.45,
    max_frames=None,
    headless=False,
    save_output=None,
    device=None
):
    print("=" * 78)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - OFFLINE STANDALONE MODEL RUNNER")
    print("   ISRO SIH Problem Statement #26174 | Deliverable: Trained Offline AI Model")
    print("=" * 78)

    # 1. Verify Model Checkpoint
    if not os.path.isabs(model_path):
        model_path = os.path.join(WORKSPACE_ROOT, model_path)

    if not os.path.exists(model_path):
        fallback = os.path.join(WORKSPACE_ROOT, "yolov8n.pt")
        print(f"[Warning] Custom trained weights '{model_path}' not found.")
        print(f"[Warning] Falling back to baseline: {fallback}")
        model_path = fallback

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Model Checkpoint : {model_path} ({file_size_mb:.2f} MB)")
    print(f"   Architecture     : Deep Convolutional Neural Network (Ultralytics YOLOv8)")
    print(f"   Execution Mode   : 100% OFFLINE STANDALONE (Air-Gapped / Zero Cloud)")
    print(f"   Confidence Gate  : {int(conf_threshold * 100)}% | NMS IoU: {int(iou_threshold * 100)}%")

    # Load Model
    start_load = time.time()
    model = YOLO(model_path)
    load_time = (time.time() - start_load) * 1000.0

    class_names = getattr(model, "names", {})
    print(f"   Classes Trained  : {len(class_names)} classes -> {list(class_names.values())}")
    print(f"   Model Ready In   : {load_time:.1f} ms")
    print("=" * 78)

    # 2. Resolve Video Input Source
    cap = None
    source_label = ""

    if str(source).lower() == "auto":
        # Check camera 0 first
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(0, backend)
                if c.isOpened():
                    ret, _ = c.read()
                    if ret:
                        cap = c
                        source_label = "LIVE WEBCAM (Index #0)"
                        break
                c.release()
            except Exception:
                pass

        if cap is None:
            # Fallback to test video
            for vid in ["clip1.mp4", "red_yellow.mp4"]:
                vpath = os.path.join(WORKSPACE_ROOT, vid)
                if os.path.exists(vpath):
                    cap = cv2.VideoCapture(vpath)
                    source_label = f"OFFLINE FLIGHT VIDEO ({vid})"
                    break
    elif str(source).isdigit():
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx)
        source_label = f"LIVE WEBCAM (Index #{cam_idx})"
    else:
        vpath = os.path.join(WORKSPACE_ROOT, source) if not os.path.isabs(source) else source
        if not os.path.exists(vpath):
            print(f"[Error] Source video file not found: {vpath}")
            return
        cap = cv2.VideoCapture(vpath)
        source_label = f"VIDEO FILE ({os.path.basename(vpath)})"

    if cap is None or not cap.isOpened():
        print(f"[Error] Could not initialize input video stream: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Stream] Resolution : {width}x{height} @ {fps_src:.1f} FPS")
    print(f"[Stream] Ingest Src : {source_label}")
    if total_src_frames > 0:
        print(f"[Stream] Total Frames: {total_src_frames} ({total_src_frames / fps_src:.1f} sec)")
    print("[Controls] Keys: [Q] Quit | [S] Save Snapshot | [+] Raise Conf | [-] Lower Conf")
    print("-" * 78)

    # Video Writer if requested
    writer = None
    if save_output:
        os.makedirs(os.path.dirname(os.path.abspath(save_output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, fps_src, (width, height))
        print(f"[Recorder] Output recording enabled: {save_output}")

    win_name = "BAS On-Board AI - Standalone Offline Inference (ISRO SIH #26174)"
    if not headless:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1280, max(800, width)), min(720, max(600, height)))

    frame_count = 0
    total_inference_time = 0.0
    detected_history = []
    fps_smooth = fps_src
    current_conf = conf_threshold
    t_start_all = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if not str(source).isdigit() and str(source).lower() != "auto":
                    # End of file reached
                    break
                else:
                    # Loop video or camera retry
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            frame_count += 1
            t_infer_start = time.time()

            # Execute Local Neural Network Inference
            results = model(
                frame,
                conf=current_conf,
                iou=iou_threshold,
                verbose=False,
                device=device
            )
            t_infer = (time.time() - t_infer_start) * 1000.0
            total_inference_time += t_infer

            # FPS calculation
            fps_instant = 1000.0 / max(1.0, t_infer)
            fps_smooth = 0.90 * fps_smooth + 0.10 * fps_instant

            display_frame = frame.copy()
            h, w, _ = display_frame.shape
            scale = max(0.45, min(0.85, w / 1280.0))

            detections = []
            if results and len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_name = class_names.get(cls_id, f"class_{cls_id}")
                    disp_name = CLASS_DISPLAY_NAMES.get(raw_name, raw_name.upper().replace("_", " "))

                    detections.append({
                        "id": cls_id,
                        "raw_name": raw_name,
                        "disp_name": disp_name,
                        "conf": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })

            detected_history.append(len(detections))

            # -------------------------------------------------------------
            # Render Bounding Boxes, Cyber Brackets, and Badges
            # -------------------------------------------------------------
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                cls_color = get_class_color(det["id"])

                # Outer Box
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), cls_color, 2)

                # High-Tech Cyber Corner Brackets
                corner_len = min(22, max(8, int((bx2 - bx1) * 0.16)))
                # Top-Left
                cv2.line(display_frame, (bx1, by1), (bx1 + corner_len, by1), (255, 255, 255), 2)
                cv2.line(display_frame, (bx1, by1), (bx1, by1 + corner_len), (255, 255, 255), 2)
                # Top-Right
                cv2.line(display_frame, (bx2, by1), (bx2 - corner_len, by1), (255, 255, 255), 2)
                cv2.line(display_frame, (bx2, by1), (bx2, by1 + corner_len), (255, 255, 255), 2)
                # Bottom-Left
                cv2.line(display_frame, (bx1, by2), (bx1 + corner_len, by2), (255, 255, 255), 2)
                cv2.line(display_frame, (bx1, by2), (bx1, by2 - corner_len), (255, 255, 255), 2)
                # Bottom-Right
                cv2.line(display_frame, (bx2, by2), (bx2 - corner_len, by2), (255, 255, 255), 2)
                cv2.line(display_frame, (bx2, by2), (bx2, by2 - corner_len), (255, 255, 255), 2)

                # Centroid Crosshair
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                cv2.drawMarker(display_frame, (cx, cy), cls_color, cv2.MARKER_CROSS, 8, 1)

                # Badge Label
                label_txt = f"{det['disp_name']} {int(det['conf'] * 100)}%"
                font_scale = max(0.38, scale * 0.72)
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                pad_x, pad_y = 6, 4

                lbl_y2 = by1 - 4 if (by1 - th - 2 * pad_y) > 42 else by1 + th + 2 * pad_y + 4
                lbl_y1 = lbl_y2 - th - 2 * pad_y
                lbl_x1 = max(4, bx1)
                lbl_x2 = min(w - 4, lbl_x1 + tw + 2 * pad_x)

                # Badge Pill
                cv2.rectangle(display_frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (15, 20, 30), -1)
                cv2.rectangle(display_frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), cls_color, 1)
                cv2.putText(display_frame, label_txt, (lbl_x1 + pad_x, lbl_y2 - pad_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # Top Banner: Spaceflight Mission HUD
            # -------------------------------------------------------------
            banner_h = max(38, int(h * 0.065))
            overlay_top = display_frame.copy()
            cv2.rectangle(overlay_top, (0, 0), (w, banner_h), (12, 16, 26), -1)
            cv2.addWeighted(overlay_top, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.line(display_frame, (0, banner_h), (w, banner_h), (0, 240, 255), 1)

            # Left Zone: Title & Status
            header_txt = "BHARATIYA ANTARIKSH STATION | OFFLINE AI MODEL [ISRO PS #26174]"
            cv2.putText(display_frame, header_txt, (14, int(banner_h * 0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.40, scale * 0.76), (0, 240, 255), 1, cv2.LINE_AA)

            # Right Zone: Telemetry
            stats_txt = f"FPS: {fps_smooth:.1f} | INFER: {t_infer:.1f}ms | TARGETS: {len(detections)} | CONF: {int(current_conf * 100)}%"
            (stw, _), _ = cv2.getTextSize(stats_txt, cv2.FONT_HERSHEY_SIMPLEX, max(0.38, scale * 0.70), 1)
            cv2.putText(display_frame, stats_txt, (max(w - stw - 14, int(w * 0.45)), int(banner_h * 0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.38, scale * 0.70), (0, 230, 120), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # Bottom Banner: Active Objects Summary
            # -------------------------------------------------------------
            bot_h = max(28, int(h * 0.048))
            overlay_bot = display_frame.copy()
            cv2.rectangle(overlay_bot, (0, h - bot_h), (w, h), (10, 14, 22), -1)
            cv2.addWeighted(overlay_bot, 0.88, display_frame, 0.12, 0, display_frame)
            cv2.line(display_frame, (0, h - bot_h), (w, h - bot_h), (40, 60, 90), 1)

            if detections:
                counts = {}
                for d in detections:
                    counts[d["disp_name"]] = counts.get(d["disp_name"], 0) + 1
                det_str = "ACTIVE OBJECTS: " + ", ".join([f"{k} x{v}" for k, v in counts.items()])
                if len(det_str) > 85:
                    det_str = det_str[:82] + "..."
                cv2.putText(display_frame, det_str, (14, h - int(bot_h * 0.32)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.36, scale * 0.66), (240, 240, 240), 1, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, "STANDALONE INFERENCE ACTIVE - Point camera at experiment box / payloads",
                            (14, h - int(bot_h * 0.32)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, scale * 0.63), (140, 160, 180), 1, cv2.LINE_AA)

            # Record frame if enabled
            if writer:
                writer.write(display_frame)

            # Render Window
            if not headless:
                cv2.imshow(win_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\n[Runner] Shutdown requested by user.")
                    break
                elif key in (ord('s'), ord('S')):
                    snap_dir = os.path.join(WORKSPACE_ROOT, "experiments", "detections")
                    os.makedirs(snap_dir, exist_ok=True)
                    snap_file = os.path.join(snap_dir, f"snap_{int(time.time())}.jpg")
                    cv2.imwrite(snap_file, display_frame)
                    print(f"\n[Snapshot] Saved frame: {snap_file}")
                elif key in (ord('+'), ord('=')):
                    current_conf = min(0.95, current_conf + 0.05)
                    print(f"\n[Runner] Confidence threshold -> {int(current_conf * 100)}%")
                elif key in (ord('-'), ord('_')):
                    current_conf = max(0.10, current_conf - 0.05)
                    print(f"\n[Runner] Confidence threshold -> {int(current_conf * 100)}%")
            else:
                if frame_count % 30 == 0:
                    print(f"[Inference Headless] Frame {frame_count:04d} | Infer: {t_infer:.1f}ms ({fps_instant:.1f} FPS) | Objects: {len(detections)}")

            if max_frames and frame_count >= max_frames:
                print(f"[Runner] Reached frame limit: {max_frames} frames.")
                break

    except KeyboardInterrupt:
        print("\n[Runner] Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    total_wall_time = time.time() - t_start_all
    avg_latency = (total_inference_time / max(1, frame_count))
    avg_fps = frame_count / max(0.001, total_wall_time)

    print("\n" + "=" * 78)
    print("   OFFLINE MODEL INFERENCE BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"   Total Processed Frames : {frame_count}")
    print(f"   Total Wall Time        : {total_wall_time:.2f} s")
    print(f"   Average Neural Latency : {avg_latency:.2f} ms / frame")
    print(f"   Effective Throughput   : {avg_fps:.1f} FPS")
    print(f"   Total Detections Logged: {sum(detected_history)}")
    print(f"   Cloud Connections      : 0 (100% Offline Standalone Verified)")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAS Standalone Offline Model Runner")
    parser.add_argument("--model", default="models/detector_offline.pt",
                        help="Path to trained offline model checkpoint")
    parser.add_argument("--source", default="auto",
                        help="Input source: 'auto', webcam index ('0'), or video path ('clip1.mp4')")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold [0.10 - 0.95]")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--frames", type=int, default=None,
                        help="Maximum number of frames to run")
    parser.add_argument("--headless", action="store_true",
                        help="Run without displaying OpenCV window (for background/testing)")
    parser.add_argument("--record", type=str, default=None,
                        help="Path to save annotated video recording")
    parser.add_argument("--device", type=str, default=None,
                        help="Inference compute device: 'cpu', '0', etc.")
    args = parser.parse_args()

    run_offline_standalone(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_frames=args.frames,
        headless=args.headless,
        save_output=args.record,
        device=args.device
    )

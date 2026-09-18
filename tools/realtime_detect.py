"""
BAS Autonomous HAR System - Real-Time Common Object Detection
Instant, standalone real-time common object detection on physical webcam.
Utilizes YOLOv8 (80 COCO common object classes: person, cell phone, bottle, cup,
laptop, mouse, keyboard, book, scissors, chair, backpack, etc.) with real-time HUD overlays.
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
    print("[Error] Ultralytics YOLO is required. Run using virtual environment:")
    print(r"  .\.venv\Scripts\python.exe realtime_detect.py")
    sys.exit(1)


# Color palette for 80 COCO classes (distinct RGB colors)
np.random.seed(42)
CLASS_COLORS = {}


def get_color(class_id: int):
    if class_id not in CLASS_COLORS:
        # Generate vivid, high-visibility colors in BGR
        hue = int((class_id * 37) % 180)
        hsv_pixel = np.uint8([[[hue, 230, 240]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        CLASS_COLORS[class_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return CLASS_COLORS[class_id]


def run_realtime_detection(
    source="0",
    model_path=None,
    conf_threshold=0.30,
    iou_threshold=0.45,
    save_dir="experiments/detections",
    max_frames=None,
    headless=False,
    experiment_only=True
):
    if model_path is None:
        model_path = "models/detector_offline.pt" if os.path.exists("models/detector_offline.pt") else "yolov8n.pt"

    print("=" * 75)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - REAL-TIME EXPERIMENT DETECTOR")
    print("   Architecture     : 100% Offline Standalone Neural Inference (Zero Cloud)")
    print(f"   Model Weights    : {model_path}")
    print(f"   Mode             : {'EXPERIMENT OBJECTS ONLY (Clutter Filtered)' if experiment_only else 'ALL OBJECTS'}")
    print(f"   Confidence Gate  : {int(conf_threshold * 100)}%")
    print(f"   Target Video Src : Camera index #{source}" if str(source).isdigit() else f"   Target Video Src : {source}")
    print("=" * 75)

    if not os.path.exists(model_path):
        print(f"[Model] Checkpoint '{model_path}' not found locally. Loading yolov8n.pt...")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)

    print(f"[Model] Successfully initialized offline detector with {len(model.names)} classes: {list(model.names.values())}")

    # 1. Initialize Video Source
    cap = None
    source_name = "LIVE WEBCAM"
    if str(source).lower() == "auto":
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(0, backend)
                if c.isOpened():
                    ret_t, _ = c.read()
                    if ret_t:
                        cap = c
                        source_name = "LIVE WEBCAM #0"
                        source = "0"
                        break
                c.release()
            except Exception:
                pass
        if cap is None:
            fallback = "clip1.mp4" if os.path.exists("clip1.mp4") else "clip.mp4"
            if os.path.exists(fallback):
                print(f"[Camera] No hardware camera accessible. Defaulting to: {fallback}")
                cap = cv2.VideoCapture(fallback)
                source = fallback
                source_name = f"FILE: {os.path.basename(fallback)}"
    elif str(source).isdigit():
        cam_idx = int(source)
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(cam_idx, backend)
                if c.isOpened():
                    ret_t, _ = c.read()
                    if ret_t:
                        cap = c
                        source_name = f"LIVE WEBCAM #{cam_idx}"
                        break
                c.release()
            except Exception:
                pass

        if cap is None and cam_idx != 0:
            print(f"[Warning] Camera index #{cam_idx} unavailable. Probing Camera #0...")
            for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    c = cv2.VideoCapture(0, backend)
                    if c.isOpened():
                        ret_t, _ = c.read()
                        if ret_t:
                            cap = c
                            source_name = "LIVE WEBCAM #0"
                            break
                    c.release()
                except Exception:
                    pass
    else:
        if not os.path.exists(source):
            print(f"[Error] Source video file not found: {source}")
            return
        cap = cv2.VideoCapture(source)
        source_name = f"FILE: {os.path.basename(source)}"

    if cap is None or not cap.isOpened():
        print(f"[Error] Could not open camera device or video source: {source}")
        return

    # Real-time capture settings
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Stream ready: {cw}x{ch} px | Source: {source_name} | Buffer=1 real-time.")
    print("[Controls] Keys: [Q] Quit | [S] Save Snapshot | [+] Increase Conf | [-] Decrease Conf")
    print("=" * 75)

    os.makedirs(save_dir, exist_ok=True)

    prev_time = time.time()
    fps_smooth = 30.0
    frame_count = 0
    current_conf = conf_threshold

    win_name = "BAS - Real-Time Common Object Detection (COCO-80)"
    if not headless:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1280, max(800, cw)), min(720, max(600, ch)))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if not str(source).isdigit():
                    # Loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.01)
                    continue

            frame_count += 1
            now = time.time()
            dt = max(1e-4, now - prev_time)
            fps_instant = 1.0 / dt
            fps_smooth = 0.90 * fps_smooth + 0.10 * fps_instant
            prev_time = now

            h, w, _ = frame.shape
            scale = max(0.45, min(0.85, w / 1280.0))

            # Run YOLOv8 Common Object Inference
            results = model(frame, verbose=False, conf=current_conf, iou=iou_threshold)
            detections = []

            # Filter definitions: ONLY label experiment things
            EXPERIMENT_CONTAINERS = {"suitcase", "backpack", "handbag", "box", "container_box"}
            EXPERIMENT_PAYLOADS = {"bottle", "cup", "cell phone", "book", "bowl", "scissors", "component_box", "red_box", "yellow_box"}
            EXPERIMENT_HANDS = {"hand", "operator_hand", "astronaut_hand"}
            EXPERIMENT_LIDS = {"lid", "container_lid"}

            if results and len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_name = model.names.get(cls_id, f"obj_{cls_id}").lower()

                    if experiment_only:
                        # Strictly ignore all non-experiment background clutter (person, chair, laptop, desk, etc.)
                        if raw_name in EXPERIMENT_CONTAINERS:
                            disp_name = "CONTAINER BOX"
                        elif raw_name in EXPERIMENT_PAYLOADS:
                            disp_name = f"PAYLOAD ({raw_name.upper()})" if raw_name not in ("component_box", "red_box", "yellow_box") else raw_name.upper().replace("_", " ")
                        elif raw_name in EXPERIMENT_HANDS:
                            disp_name = "OPERATOR HAND"
                        elif raw_name in EXPERIMENT_LIDS:
                            disp_name = "CONTAINER LID"
                        else:
                            # Skip this background object!
                            continue
                    else:
                        disp_name = raw_name.upper()

                    detections.append({
                        "id": cls_id,
                        "name": disp_name,
                        "conf": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })

            display_frame = frame.copy()

            # Render Bounding Boxes and Floating Badges
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                cls_color = get_color(det["id"])
                
                # Main bounding box outline
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), cls_color, 2)

                # Corner accent brackets (high-tech cyber look)
                corner_len = min(20, max(8, int((bx2 - bx1) * 0.15)))
                # Top-Left
                cv2.line(display_frame, (bx1, by1), (bx1 + corner_len, by1), (255, 255, 255), 3)
                cv2.line(display_frame, (bx1, by1), (bx1, by1 + corner_len), (255, 255, 255), 3)
                # Top-Right
                cv2.line(display_frame, (bx2, by1), (bx2 - corner_len, by1), (255, 255, 255), 3)
                cv2.line(display_frame, (bx2, by1), (bx2, by1 + corner_len), (255, 255, 255), 3)
                # Bottom-Left
                cv2.line(display_frame, (bx1, by2), (bx1 + corner_len, by2), (255, 255, 255), 3)
                cv2.line(display_frame, (bx1, by2), (bx1, by2 - corner_len), (255, 255, 255), 3)
                # Bottom-Right
                cv2.line(display_frame, (bx2, by2), (bx2 - corner_len, by2), (255, 255, 255), 3)
                cv2.line(display_frame, (bx2, by2), (bx2, by2 - corner_len), (255, 255, 255), 3)

                # Centroid crosshair
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                cv2.drawMarker(display_frame, (cx, cy), cls_color, cv2.MARKER_CROSS, 8, 1)

                # Pill Label Badge
                label_txt = f"{det['name']} {int(det['conf'] * 100)}%"
                font_scale = max(0.38, scale * 0.75)
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                pad_x, pad_y = 6, 4

                # Place badge above box, or flip inside if near screen top
                lbl_y2 = by1 - 4 if (by1 - th - 2 * pad_y) > 45 else by1 + th + 2 * pad_y + 4
                lbl_y1 = lbl_y2 - th - 2 * pad_y
                lbl_x1 = max(4, bx1)
                lbl_x2 = min(w - 4, lbl_x1 + tw + 2 * pad_x)

                # Filled background pill
                cv2.rectangle(display_frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (18, 22, 32), -1)
                cv2.rectangle(display_frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), cls_color, 1)

                # Text
                text_y = lbl_y2 - pad_y
                cv2.putText(display_frame, label_txt, (lbl_x1 + pad_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # Top Header Bar (Semi-transparent HUD)
            banner_h = max(38, int(h * 0.065))
            overlay_top = display_frame.copy()
            cv2.rectangle(overlay_top, (0, 0), (w, banner_h), (12, 16, 26), -1)
            cv2.addWeighted(overlay_top, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.line(display_frame, (0, banner_h), (w, banner_h), (0, 240, 255), 1)

            # Header Zone 1 (Left): Mission & Filter Status
            mode_tag = "EXPERIMENT ITEMS ONLY" if experiment_only else "ALL OBJECTS"
            title_txt = f"BAS REAL-TIME DETECTOR | {mode_tag} [KEY E]"
            cv2.putText(display_frame, title_txt, (14, int(banner_h * 0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.40, scale * 0.78), (0, 240, 255), 1, cv2.LINE_AA)

            # Header Zone 2 (Center/Right): FPS & Detection Stats
            stats_txt = f"FPS: {fps_smooth:.1f} | EXPERIMENT ITEMS: {len(detections)} | CONF: {int(current_conf * 100)}%"
            (stw, _), _ = cv2.getTextSize(stats_txt, cv2.FONT_HERSHEY_SIMPLEX, max(0.38, scale * 0.72), 1)
            stats_x = max(w - stw - 16, int(w * 0.48))
            cv2.putText(display_frame, stats_txt, (stats_x, int(banner_h * 0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.38, scale * 0.72), (0, 230, 118), 1, cv2.LINE_AA)

            # Bottom Footer Bar
            bot_h = max(28, int(h * 0.048))
            overlay_bot = display_frame.copy()
            cv2.rectangle(overlay_bot, (0, h - bot_h), (w, h), (10, 14, 22), -1)
            cv2.addWeighted(overlay_bot, 0.88, display_frame, 0.12, 0, display_frame)
            cv2.line(display_frame, (0, h - bot_h), (w, h - bot_h), (40, 60, 90), 1)

            # Detected summary in footer
            if detections:
                unique_counts = {}
                for d in detections:
                    unique_counts[d["name"]] = unique_counts.get(d["name"], 0) + 1
                det_summary = "EXPERIMENT ITEMS: " + ", ".join([f"{k} x{v}" for k, v in unique_counts.items()])
                if len(det_summary) > 75:
                    det_summary = det_summary[:72] + "..."
                cv2.putText(display_frame, det_summary, (14, h - int(bot_h * 0.32)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.38, scale * 0.68), (240, 240, 240), 1, cv2.LINE_AA)
            else:
                foot_msg = "EXPERIMENT ONLY: Container, Box, Payloads & Hands labeled. (Person/Chair clutter hidden)." if experiment_only else "ALL OBJECTS: Point camera at any object."
                cv2.putText(display_frame, foot_msg,
                            (14, h - int(bot_h * 0.32)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, scale * 0.63), (140, 160, 180), 1, cv2.LINE_AA)

            # Show window if not headless
            if not headless:
                cv2.imshow(win_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                    print("\n[Detection] Exit requested by user.")
                    break
                elif key in (ord('e'), ord('E')):
                    experiment_only = not experiment_only
                    print(f"\n[Detection] Mode switched to: {'EXPERIMENT OBJECTS ONLY' if experiment_only else 'ALL OBJECTS'}")
                elif key in (ord('s'), ord('S')):
                    snap_path = os.path.join(save_dir, f"detection_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, display_frame)
                    print(f"\n[Snapshot] Saved detection screenshot: {snap_path}")
                elif key in (ord('+'), ord('=')):
                    current_conf = min(0.95, current_conf + 0.05)
                    print(f"\n[Detection] Confidence threshold raised to: {int(current_conf * 100)}%")
                elif key in (ord('-'), ord('_')):
                    current_conf = max(0.10, current_conf - 0.05)
                    print(f"\n[Detection] Confidence threshold lowered to: {int(current_conf * 100)}%")
            else:
                time.sleep(0.001)

            if max_frames is not None and frame_count >= max_frames:
                print(f"[Detection] Completed max_frames target ({max_frames} frames).")
                break

    except KeyboardInterrupt:
        print("\n[Detection] Shutdown requested.")
    finally:
        if cap:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"[Detection] Camera session closed. Total frames processed: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAS Real-Time Experiment Object Detector")
    parser.add_argument("--source", default="auto",
                        help="Camera index ('0'), 'auto' (probes webcam #0 first), or path to video file (default: auto)")
    default_weights = "models/detector_offline.pt" if os.path.exists("models/detector_offline.pt") else "yolov8n.pt"
    parser.add_argument("--model", default=default_weights,
                        help=f"YOLO model checkpoint (default: {default_weights})")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Confidence threshold [0.1 - 0.95] (default: 0.30)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Maximum frames to process (optional)")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless without opening GUI display window")
    parser.add_argument("--all", action="store_true",
                        help="Disable experiment-only filter and label all 80 COCO objects")
    args = parser.parse_args()

    run_realtime_detection(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        max_frames=args.frames,
        headless=args.headless,
        experiment_only=not args.all
    )

"""
BAS Autonomous HAR System - 3D Digital Twin Dataset Feeder & Player
Ingests offline datasets (har_dataset_3d_joints.json, har_dataset_telemetry.csv)
and plays back the 3D scene in the Digital Twin in real-time, synchronizing:
- Payload Rack Frame R workbench
- Astronaut full skeleton rig & active hand coordinates
- Container box & animated lid opening/closing
- Component extraction, grasp, and dock transitions
- Live HAR activity labels and FSM procedural steps
"""

import os
import sys
import csv
import json
import time
import argparse
import cv2
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.agents.digital_twin_agent import DigitalTwinAgent


def load_dataset_records(dataset_path: str):
    """Loads dataset frames from JSON or CSV format."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "records" in data:
            return data["records"], "json"
        elif isinstance(data, list):
            return data, "json"
        else:
            raise ValueError("Unrecognized JSON dataset structure (must contain 'records' list).")

    elif dataset_path.endswith(".csv"):
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records, "csv"

    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path} (use .json or .csv)")


def is_display_available() -> bool:
    """Checks if a graphical display environment is available."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def play_dataset_in_digital_twin(
    dataset_path: str = "dataset/har_dataset/har_dataset_3d_joints.json",
    playback_fps: float = 24.0,
    loop: bool = False,
    max_frames: int = None,
    save_video_path: str = None,
    show_window: bool = True,
    canvas_width: int = 640,
    canvas_height: int = 480
):
    print("=" * 75)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - DIGITAL TWIN DATASET FEEDER")
    print(f"   Input Dataset  : {dataset_path}")
    print(f"   Playback Speed : {playback_fps:.1f} FPS")
    print(f"   Canvas Size    : {canvas_width}x{canvas_height}")
    if save_video_path:
        print(f"   Export Video   : {save_video_path}")
    print("=" * 75)

    if show_window and not is_display_available():
        print("[Feeder] Notice: No graphical display detected (headless environment). Running in headless mode.")
        show_window = False

    records, fmt = load_dataset_records(dataset_path)
    total_records = len(records)
    print(f"[Feeder] Loaded {total_records} frame records from {fmt.upper()} dataset.")

    if total_records == 0:
        print("[Feeder] Warning: Dataset is empty.")
        return

    agent_twin = DigitalTwinAgent()
    video_writer = None
    if save_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_video_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video_path, fourcc, playback_fps, (canvas_width, canvas_height))

    frame_delay = max(0.001, 1.0 / playback_fps)
    idx = 0
    paused = False
    print("\n[Controls] SPACE: Pause/Play | LEFT/RIGHT: Step frame | R: Restart | Q: Quit\n")

    try:
        while True:
            if idx >= total_records:
                if loop:
                    idx = 0
                    print("[Feeder] Looping dataset from Frame 0...")
                else:
                    print("[Feeder] Reached end of dataset playback.")
                    break

            if max_frames and idx >= max_frames:
                break

            record = records[idx]
            t_frame_start = time.time()

            # Feed dataset record into Digital Twin
            scene_graph = agent_twin.sync_from_dataset_record(record)
            canvas = agent_twin.render_digital_twin_canvas(
                scene_graph, width=canvas_width, height=canvas_height
            )

            # Overlay Playback Telemetry HUD Header
            frame_id = record.get("frame_id", idx + 1)
            time_sec = record.get("timestamp_sec", round(idx * frame_delay, 2))
            header_text = f"DATASET FEED | FRAME: {frame_id}/{total_records} | T: {time_sec}s | {playback_fps:.1f} FPS"
            cv2.putText(canvas, header_text, (canvas_width - 340, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1)

            if video_writer:
                video_writer.write(canvas)

            if show_window:
                cv2.imshow("BAS 3D Digital Twin - Live Dataset Playback", canvas)
                key = cv2.waitKey(int(frame_delay * 1000) if not paused else 50) & 0xFF

                if key == ord('q') or key == 27:
                    print("[Feeder] Exit requested by user.")
                    break
                elif key == 32:  # SPACE
                    paused = not paused
                    print(f"[Feeder] {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):
                    idx = 0
                    print("[Feeder] Reset playback to frame 0.")
                    continue
                elif key in (ord('d'), 83):  # Next frame
                    idx = min(total_records - 1, idx + 1)
                    continue
                elif key in (ord('a'), 81):  # Previous frame
                    idx = max(0, idx - 1)
                    continue

            if not paused:
                idx += 1
                elapsed = time.time() - t_frame_start
                sleep_time = max(0.0, frame_delay - elapsed)
                if sleep_time > 0 and not show_window:
                    time.sleep(sleep_time)

    finally:
        if video_writer:
            video_writer.release()
            print(f"[Feeder] Exported Digital Twin playback video: {save_video_path}")
        if show_window and is_display_available():
            cv2.destroyAllWindows()

    print("[Feeder] Digital Twin dataset playback finished cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed and Playback Dataset in 3D Digital Twin")
    parser.add_argument("--dataset", default="dataset/har_dataset/har_dataset_3d_joints.json",
                        help="Path to dataset file (.json or .csv)")
    parser.add_argument("--fps", type=float, default=24.0, help="Playback FPS")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to play")
    parser.add_argument("--save-video", default=None, help="Optional output MP4 video path")
    parser.add_argument("--no-window", action="store_true", help="Headless mode (no OpenCV GUI)")
    parser.add_argument("--width", type=int, default=640, help="Canvas width")
    parser.add_argument("--height", type=int, default=480, help="Canvas height")
    args = parser.parse_args()

    # Fallback to telemetry CSV if default JSON does not exist
    dataset_target = args.dataset
    if not os.path.exists(dataset_target):
        for candidate in [
            "dataset/har_dataset/har_dataset_3d_joints.json",
            "dataset/har_dataset/har_dataset_telemetry.csv",
            "experiments/latest_telemetry.csv",
            "realtime_feed/current_feed_telemetry.csv",
        ]:
            if os.path.exists(candidate):
                dataset_target = candidate
                break

    play_dataset_in_digital_twin(
        dataset_path=dataset_target,
        playback_fps=args.fps,
        loop=args.loop,
        max_frames=args.max_frames,
        save_video_path=args.save_video,
        show_window=not args.no_window and is_display_available(),
        canvas_width=args.width,
        canvas_height=args.height
    )

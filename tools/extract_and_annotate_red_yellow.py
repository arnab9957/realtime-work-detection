"""
Dataset Extractor and Auto-Annotator for red_yellow.mp4
Generates a YOLOv8 dataset with 6 classes:
0: container_box
1: container_lid
2: red_box
3: yellow_box
4: operator_hand
5: human_body
"""
import os
import sys
import cv2
import json
import random
import numpy as np

# Ensure workspace root in sys.path
WORKSPACE_ROOT = r"e:\BAS\realtime-work-detection"
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ultralytics import YOLO


def find_colored_box(hsv, lower1, upper1, lower2=None, upper2=None, min_area=1200, roi=None):
    m1 = cv2.inRange(hsv, lower1, upper1)
    mask = m1
    if lower2 is not None and upper2 is not None:
        mask = cv2.bitwise_or(m1, cv2.inRange(hsv, lower2, upper2))
    if roi:
        rx1, ry1, rx2, ry2 = roi
        h, w = mask.shape[:2]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        r_mask = np.zeros_like(mask)
        r_mask[ry1:ry2, rx1:rx2] = mask[ry1:ry2, rx1:rx2]
        mask = r_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) >= min_area:
            x, y, w, h = cv2.boundingRect(best)
            return [x, y, x + w, y + h]
    return None


def main():
    video_path = os.path.join(WORKSPACE_ROOT, "clip.mp4")
    out_dir = os.path.join(WORKSPACE_ROOT, "dataset", "red_yellow_dataset")
    stride = 1
    val_ratio = 0.15

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    pose_model_path = os.path.join(WORKSPACE_ROOT, "models", "yolov8n-pose.pt")
    if not os.path.exists(pose_model_path):
        pose_model_path = "yolov8n-pose.pt"

    pose_model = YOLO(pose_model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Annotator] Processing {video_path} ({total_frames} frames @ {fps:.1f} FPS, stride={stride})...")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            sec = frame_idx / fps
            is_val = (random.random() < val_ratio)
            split = "val" if is_val else "train"
            stem = f"frame_{frame_idx:05d}"
            img_path = os.path.join(out_dir, "images", split, f"{stem}.jpg")
            lbl_path = os.path.join(out_dir, "labels", split, f"{stem}.txt")

            boxes = []  # (cls_id, x1, y1, x2, y2)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 1. Pose Model for Human Body and Hands
            pose_res = list(pose_model(frame, verbose=False))[0]  # type: ignore
            if getattr(pose_res, 'boxes', None) is not None and pose_res.boxes:  # type: ignore
                # Largest person box
                pb = max(pose_res.boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))  # type: ignore
                px1, py1, px2, py2 = [float(c) for c in pb.xyxy[0].tolist()]
                boxes.append((5, px1, py1, px2, py2))  # human_body

            # Wrists for operator hands
            if getattr(pose_res, 'keypoints', None) is not None and len(pose_res.keypoints) > 0:  # type: ignore
                kps = pose_res.keypoints[0].data[0].tolist()  # type: ignore
                for wid in [9, 10]:
                    if wid < len(kps) and kps[wid][2] > 0.3:
                        wx, wy = kps[wid][0], kps[wid][1]
                        hw = 40
                        boxes.append((4, max(0, wx - hw), max(0, wy - hw), min(w_orig, wx + hw), min(h_orig, wy + hw)))

            # 2. Container Box & Lid
            cb_x1, cb_y1, cb_x2, cb_y2 = 275, 325, 495, 455
            boxes.append((0, cb_x1, cb_y1, cb_x2, cb_y2))  # container_box

            # Lid is open from ~3.5s to ~22.0s
            if 3.5 <= sec <= 22.0:
                lid_box = (1, 260, 275, 515, 365)  # container_lid (open flaps)
                boxes.append(lid_box)
            else:
                lid_box = (1, 285, 320, 485, 360)  # container_lid (closed)
                boxes.append(lid_box)

            # 3. Red Box: Inside container from 3.5s to 4.5s, extracted 4.5s to 11.5s, on floor 11.5s to 20.5s
            r_box = None
            if 3.5 <= sec < 4.5:
                # Inside container
                r_box = find_colored_box(
                    hsv,
                    np.array([0, 75, 55]), np.array([12, 255, 255]),
                    np.array([160, 75, 55]), np.array([180, 255, 255]),
                    min_area=500, roi=(275, 325, 495, 455)
                )
            elif 4.5 <= sec <= 11.5:
                r_box = find_colored_box(
                    hsv,
                    np.array([0, 75, 55]), np.array([12, 255, 255]),
                    np.array([160, 75, 55]), np.array([180, 255, 255]),
                    min_area=2500, roi=(250, 70, 550, 320)
                )
            elif 11.5 < sec <= 20.5:
                r_box = find_colored_box(
                    hsv,
                    np.array([0, 75, 55]), np.array([12, 255, 255]),
                    np.array([160, 75, 55]), np.array([180, 255, 255]),
                    min_area=1500, roi=(120, 280, 290, 460)
                )
            if r_box:
                boxes.append((2, r_box[0], r_box[1], r_box[2], r_box[3]))  # red_box

            # 4. Yellow Box: Inside container 3.5s to 11.0s, extracted 11.0s to 17.5s, returned 17.5s to 21.0s
            y_box = None
            if 3.5 <= sec < 11.0:
                # Inside container
                y_box = find_colored_box(
                    hsv,
                    np.array([20, 75, 75]), np.array([36, 255, 255]),
                    min_area=500, roi=(275, 325, 495, 455)
                )
            elif 11.0 <= sec <= 17.5:
                y_box = find_colored_box(
                    hsv,
                    np.array([20, 75, 75]), np.array([36, 255, 255]),
                    min_area=1500, roi=(320, 80, 520, 300)
                )
            elif 17.5 < sec <= 21.0:
                y_box = find_colored_box(
                    hsv,
                    np.array([20, 75, 75]), np.array([36, 255, 255]),
                    min_area=1200, roi=(270, 250, 490, 380)
                )
            if y_box:
                boxes.append((3, y_box[0], y_box[1], y_box[2], y_box[3]))  # yellow_box

            # Save image
            cv2.imwrite(img_path, frame)

            # Write YOLO normalized annotations
            with open(lbl_path, "w") as f:
                for cls_id, bx1, by1, bx2, by2 in boxes:
                    bw = max(2.0, bx2 - bx1)
                    bh = max(2.0, by2 - by1)
                    cx = bx1 + bw / 2.0
                    cy = by1 + bh / 2.0
                    f.write(f"{cls_id} {cx/w_orig:.6f} {cy/h_orig:.6f} {bw/w_orig:.6f} {bh/h_orig:.6f}\n")

            saved += 1

        frame_idx += 1

    cap.release()

    yaml_path = os.path.join(out_dir, "data.yaml")
    yaml_content = f"""path: {os.path.abspath(out_dir).replace('\\', '/')}
train: images/train
val: images/val

names:
  0: container_box
  1: container_lid
  2: red_box
  3: yellow_box
  4: operator_hand
  5: human_body
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[Annotator] Done! Saved {saved} annotated frames to {out_dir}")
    print(f"[Annotator] YAML created at {yaml_path}")


if __name__ == "__main__":
    random.seed(42)
    main()

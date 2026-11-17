"""
Water Bottle Experiment - Synthetic Dataset & Flight Demonstration Generator
Generates:
1. YOLO26 Object Detection Dataset (bottle, bottle_cap, operator_hand, human_body)
2. 10D Biomechanical Kinematic Telemetry Dataset for Procedural HAR
3. Ground-truth annotated simulation video (demo_bottle.mp4)
"""

import os
import sys
import math
import random
import csv
import cv2
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(WORKSPACE_ROOT, "dataset", "water_bottle_dataset")


def draw_water_bottle(img, center, height, width, cap_attached=True, tilt_deg=0.0):
    """
    Renders realistic semi-transparent water bottle with cap, water fill, and neck.
    Returns:
        bottle_bbox: [x1, y1, x2, y2]
        cap_bbox: [x1, y1, x2, y2] or None
        top_rim_pt: (x, y)
    """
    cx, cy = center
    rad = math.radians(tilt_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    # Dimensions
    body_h = int(height * 0.70)
    neck_h = int(height * 0.20)
    cap_h = int(height * 0.10)
    body_w = width
    neck_w = int(width * 0.45)

    # 4 points for body in local frame
    local_body = [
        (-body_w // 2, body_h // 2),
        (body_w // 2, body_h // 2),
        (body_w // 2, -body_h // 2),
        (-body_w // 2, -body_h // 2),
    ]

    # Transform body points
    pts_body = []
    for lx, ly in local_body:
        rx = int(cx + lx * cos_a - ly * sin_a)
        ry = int(cy + lx * sin_a + ly * cos_a)
        pts_body.append([rx, ry])
    pts_body = np.array(pts_body, np.int32)

    # Draw Bottle Body (Cyan/Translucent glass look)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_body], (230, 215, 140))  # Light cyan/blue water
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.polylines(img, [pts_body], True, (255, 240, 180), 2, cv2.LINE_AA)

    # Bottle Neck
    local_neck = [
        (-neck_w // 2, -body_h // 2),
        (neck_w // 2, -body_h // 2),
        (neck_w // 2, -body_h // 2 - neck_h),
        (-neck_w // 2, -body_h // 2 - neck_h),
    ]
    pts_neck = []
    for lx, ly in local_neck:
        rx = int(cx + lx * cos_a - ly * sin_a)
        ry = int(cy + lx * sin_a + ly * cos_a)
        pts_neck.append([rx, ry])
    pts_neck = np.array(pts_neck, np.int32)
    cv2.fillPoly(img, [pts_neck], (240, 230, 190))
    cv2.polylines(img, [pts_neck], True, (255, 240, 180), 2, cv2.LINE_AA)

    # Top rim coordinates
    top_x = int(cx + 0 * cos_a - (-body_h // 2 - neck_h) * sin_a)
    top_y = int(cy + 0 * sin_a + (-body_h // 2 - neck_h) * cos_a)
    top_rim_pt = (top_x, top_y)

    cap_bbox = None
    if cap_attached:
        local_cap = [
            (-neck_w // 2 - 2, -body_h // 2 - neck_h),
            (neck_w // 2 + 2, -body_h // 2 - neck_h),
            (neck_w // 2 + 2, -body_h // 2 - neck_h - cap_h),
            (-neck_w // 2 - 2, -body_h // 2 - neck_h - cap_h),
        ]
        pts_cap = []
        for lx, ly in local_cap:
            rx = int(cx + lx * cos_a - ly * sin_a)
            ry = int(cy + lx * sin_a + ly * cos_a)
            pts_cap.append([rx, ry])
        pts_cap = np.array(pts_cap, np.int32)
        # Deep Blue cap
        cv2.fillPoly(img, [pts_cap], (180, 50, 20))
        cv2.polylines(img, [pts_cap], True, (220, 100, 50), 2, cv2.LINE_AA)

        c_x1 = max(0, min(pts_cap[:, 0]))
        c_y1 = max(0, min(pts_cap[:, 1]))
        c_x2 = min(img.shape[1], max(pts_cap[:, 0]))
        c_y2 = min(img.shape[0], max(pts_cap[:, 1]))
        cap_bbox = [c_x1, c_y1, c_x2, c_y2]

    # Overall bottle bbox
    all_pts = np.vstack([pts_body, pts_neck])
    b_x1 = max(0, min(all_pts[:, 0]))
    b_y1 = max(0, min(all_pts[:, 1]))
    b_x2 = min(img.shape[1], max(all_pts[:, 0]))
    b_y2 = min(img.shape[0], max(all_pts[:, 1]))
    bottle_bbox = [b_x1, b_y1, b_x2, b_y2]

    return bottle_bbox, cap_bbox, top_rim_pt


def draw_human_operator(img, mouth_pos, hand_l, hand_r):
    """Draws astronaut / operator silhouette, head, arms, and hands."""
    mx, my = mouth_pos
    # Head
    cv2.circle(img, (mx, my - 25), 38, (190, 195, 205), -1, cv2.LINE_AA)
    cv2.circle(img, (mx, my - 25), 38, (120, 130, 145), 2, cv2.LINE_AA)
    # Mouth line
    cv2.line(img, (mx - 10, my), (mx + 10, my), (70, 70, 150), 2, cv2.LINE_AA)
    # Eyes
    cv2.circle(img, (mx - 12, my - 32), 4, (40, 40, 40), -1)
    cv2.circle(img, (mx + 12, my - 32), 4, (40, 40, 40), -1)

    # Torso
    torso_pts = np.array([
        [mx - 75, my + 45],
        [mx + 75, my + 45],
        [mx + 90, img.shape[0]],
        [mx - 90, img.shape[0]]
    ], np.int32)
    cv2.fillPoly(img, [torso_pts], (65, 75, 95))
    cv2.polylines(img, [torso_pts], True, (95, 115, 145), 2, cv2.LINE_AA)

    # Hands
    hlx, hly = hand_l
    hrx, hry = hand_r
    cv2.circle(img, (hlx, hly), 18, (185, 200, 220), -1, cv2.LINE_AA)
    cv2.circle(img, (hlx, hly), 18, (110, 130, 160), 2, cv2.LINE_AA)
    cv2.circle(img, (hrx, hry), 18, (185, 200, 220), -1, cv2.LINE_AA)
    cv2.circle(img, (hrx, hry), 18, (110, 130, 160), 2, cv2.LINE_AA)

    # Arms connecting shoulders to hands
    cv2.line(img, (mx - 65, my + 50), (hlx, hly), (75, 85, 105), 8, cv2.LINE_AA)
    cv2.line(img, (mx + 65, my + 50), (hrx, hry), (75, 85, 105), 8, cv2.LINE_AA)

    hand_bboxes = [
        [hlx - 22, hly - 22, hlx + 22, hly + 22],
        [hrx - 22, hry - 22, hrx + 22, hry + 22]
    ]
    body_bbox = [mx - 95, my - 65, mx + 95, img.shape[0]]
    return hand_bboxes, body_bbox


def generate_experiment_dataset(num_frames=300):
    """Generates synthetic dataset and demo video for the 5-step drinking sequence."""
    print("=" * 70)
    print("   GENERATING WATER BOTTLE EXPERIMENT SYNTHETIC DATASET & DEMO VIDEO")
    print("=" * 70)

    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)

    # Video output configuration
    w, h = 1280, 720
    demo_video_path = os.path.join(DATASET_DIR, "demo_bottle.mp4")
    root_demo_path = os.path.join(WORKSPACE_ROOT, "demo_bottle.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(demo_video_path, fourcc, 30.0, (w, h))

    # CSV Telemetry File
    telemetry_csv_path = os.path.join(DATASET_DIR, "telemetry.csv")
    csv_file = open(telemetry_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "timestamp_sec", "step_id", "step_name",
        "bottle_elevation_m", "hand_to_bottle_dist_m", "hand_to_cap_dist_m",
        "bottle_to_mouth_dist_m", "bottle_pitch_deg", "cap_state",
        "elbow_angle_deg", "wrist_elevation_y", "is_bottle_contact", "head_tilt_deg"
    ])

    # Simulation timeline parameters (300 frames @ 30 FPS = 10.0 seconds)
    # Frame ranges:
    # 0 - 45   : Step 0 (IDLE) - bottle resting on table, hand approaches
    # 46 - 95  : Step 1 (BOTTLE_TAKEN) - hand grasps and elevates bottle
    # 96 - 150 : Step 2 (CAP_OPENED) - secondary hand unscrews and detaches cap
    # 151 - 215: Step 3 (DRINKING) - bottle tilted to mouth, sip taken
    # 216 - 260: Step 4 (CAP_CLOSED) - cap replaced and screwed on
    # 261 - 300: Step 5 (COMPLETE) - bottle placed back on table, hand retracts

    rest_bx, rest_by = 750, 520
    mouth_x, mouth_y = 520, 260
    head_rest_pos = (mouth_x, mouth_y)

    for f in range(num_frames):
        t_sec = f / 30.0
        frame = np.full((h, w, 3), (35, 40, 48), dtype=np.uint8)

        # Table surface line
        cv2.rectangle(frame, (0, 570), (w, h), (55, 60, 72), -1)
        cv2.line(frame, (0, 570), (w, 570), (85, 95, 115), 2)

        # State Determination
        if f <= 45:
            step_id = 0
            step_name = "IDLE"
            bx, by = rest_bx, rest_by
            tilt = 0.0
            cap_attached = True
            # Left hand resting, right hand slowly approaching
            progress = f / 45.0
            hl = (400, 560)
            hr = (int(600 + progress * 130), int(560 - progress * 20))
        elif f <= 95:
            step_id = 1
            step_name = "BOTTLE_TAKEN"
            progress = (f - 46) / 50.0
            # Elevate bottle from table
            bx = int(rest_bx - progress * 80)
            by = int(rest_by - progress * 110)
            tilt = progress * 6.0
            cap_attached = True
            hr = (bx, by)
            hl = (int(420 + progress * 180), int(540 - progress * 130))
        elif f <= 150:
            step_id = 2
            step_name = "CAP_OPENED"
            progress = (f - 96) / 55.0
            bx = 670
            by = 410
            tilt = 8.0
            cap_attached = (progress < 0.35)
            hr = (bx, by)
            # Left hand unscrews and moves away with cap
            if cap_attached:
                hl = (bx - 5, by - 95)
            else:
                hl = (int(bx - 5 - (progress - 0.35) * 160), int(by - 95 + (progress - 0.35) * 40))
        elif f <= 215:
            step_id = 3
            step_name = "DRINKING"
            progress = (f - 151) / 65.0
            # Raise bottle to mouth and tilt
            sin_p = math.sin(progress * math.pi)
            bx = int(670 - sin_p * 105)
            by = int(410 - sin_p * 115)
            tilt = sin_p * 36.0
            cap_attached = False
            hr = (bx, by)
            hl = (480, 440)  # holding cap aside
        elif f <= 260:
            step_id = 4
            step_name = "CAP_CLOSED"
            progress = (f - 216) / 45.0
            bx = int(565 + progress * 135)
            by = int(295 + progress * 155)
            tilt = max(0.0, 10.0 * (1.0 - progress))
            cap_attached = (progress > 0.45)
            hr = (bx, by)
            # Left hand brings cap back
            hl = (int(480 + progress * 180), int(440 - progress * 80)) if not cap_attached else (bx - 5, by - 95)
        else:
            step_id = 5
            step_name = "COMPLETE"
            progress = (f - 261) / 39.0
            bx, by = rest_bx, rest_by
            tilt = 0.0
            cap_attached = True
            hr = (int(rest_bx - progress * 180), int(rest_by + progress * 40))
            hl = (380, 560)

        # Draw Person & Hands
        hand_bboxes, body_bbox = draw_human_operator(frame, head_rest_pos, hl, hr)

        # Draw Water Bottle
        bottle_bbox, cap_bbox, top_rim = draw_water_bottle(
            frame, (bx, by), height=170, width=54, cap_attached=cap_attached, tilt_deg=tilt
        )

        # If cap is detached, draw it in the secondary hand
        if not cap_attached:
            cx, cy = hl
            cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), (180, 50, 20), -1)
            cap_bbox = [cx - 12, cy - 12, cx + 12, cy + 12]

        # Calculate Biomechanical Features
        bottle_elev_m = max(0.0, (rest_by - by) / 500.0)
        hand_dist_m = math.hypot(hr[0] - bx, hr[1] - by) / 700.0
        cap_dist_m = math.hypot(hl[0] - bx, hl[1] - (by - 90)) / 700.0
        mouth_dist_m = math.hypot(top_rim[0] - mouth_x, top_rim[1] - mouth_y) / 700.0
        is_contact = 1 if hand_dist_m < 0.12 else 0
        cap_state = 1 if cap_attached else 0
        elbow_angle = 90.0 + min(70.0, bottle_elev_m * 180.0)
        wrist_elev = (h - hr[1]) / float(h)
        head_tilt = max(0.0, tilt * 0.45) if step_id == 3 else 0.0

        # Log Telemetry
        csv_writer.writerow([
            f, round(t_sec, 3), step_id, step_name,
            round(bottle_elev_m, 3), round(hand_dist_m, 3), round(cap_dist_m, 3),
            round(mouth_dist_m, 3), round(tilt, 1), cap_state,
            round(elbow_angle, 1), round(wrist_elev, 3), is_contact, round(head_tilt, 1)
        ])

        # Write to Video
        writer.write(frame)

        # Save Selected Frames for YOLO Training (every 2nd frame)
        if f % 2 == 0:
            split = "val" if f % 10 == 0 else "train"
            img_filename = f"bottle_frame_{f:04d}.jpg"
            img_path = os.path.join(DATASET_DIR, "images", split, img_filename)
            cv2.imwrite(img_path, frame)

            # YOLO Annotation: [class_id x_c y_c w h] (normalized)
            label_filename = f"bottle_frame_{f:04d}.txt"
            label_path = os.path.join(DATASET_DIR, "labels", split, label_filename)
            lines = []

            def to_yolo(bbox, cls_id):
                x1, y1, x2, y2 = bbox
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

            # 0: bottle
            lines.append(to_yolo(bottle_bbox, 0))
            # 1: bottle_cap
            if cap_bbox:
                lines.append(to_yolo(cap_bbox, 1))
            # 2: operator_hand
            for hb in hand_bboxes:
                lines.append(to_yolo(hb, 2))
            # 3: human_body
            lines.append(to_yolo(body_bbox, 3))

            with open(label_path, "w") as lf:
                lf.writelines(lines)

    writer.release()
    csv_file.close()

    # Create data.yaml
    data_yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: bottle
  1: bottle_cap
  2: operator_hand
  3: human_body
"""
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as yf:
        yf.write(data_yaml_content)

    # Link/Copy video to root for easy execution
    if os.path.exists(demo_video_path):
        import shutil
        shutil.copy(demo_video_path, root_demo_path)

    print(f"[SUCCESS] Generated {num_frames} frames simulation video: {demo_video_path}")
    print(f"[SUCCESS] Exported telemetry dataset: {telemetry_csv_path}")
    print(f"[SUCCESS] Exported YOLO26 dataset: {os.path.join(DATASET_DIR, 'data.yaml')}")


if __name__ == "__main__":
    generate_experiment_dataset(num_frames=300)

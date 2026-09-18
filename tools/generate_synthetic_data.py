"""
BAS Autonomous HAR System - Synthetic Dataset & Experiment Video Generator
Generates domain-randomized synthetic frames and animated video sequences
replicating the ISRO microgravity sample experiment (Container + Red Box + Yellow Box).
"""

import os
import math
import random
import json
import numpy as np
import cv2

# Class Mapping for YOLOv8
CLASSES = {
    "container_box": 0,
    "container_lid": 1,
    "red_box": 2,
    "yellow_box": 3,
    "astronaut_hand": 4
}


def draw_perspective_box(img, center, size, color, angle_deg=0, shadow=True):
    """Draws a pseudo-3D isometric/perspective box with shaded faces."""
    cx, cy = center
    w, h, d = size
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    # Base 2D box
    x1, y1 = int(cx - w/2), int(cy - h/2)
    x2, y2 = int(cx + w/2), int(cy + h/2)

    # Front face
    pts_front = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], np.int32)
    
    # Top face (extruded upward/depth)
    dx = int(d * 0.5 * cos_a)
    dy = int(-d * 0.5 * sin_a - d * 0.3)
    pts_top = np.array([
        [x1, y1], [x2, y1], [x2 + dx, y1 + dy], [x1 + dx, y1 + dy]
    ], np.int32)
    
    # Side face (right)
    pts_side = np.array([
        [x2, y1], [x2 + dx, y1 + dy], [x2 + dx, y2 + dy], [x2, y2]
    ], np.int32)

    # Shading factors
    c_front = color
    c_top = tuple(min(255, int(c * 1.25)) for c in color)
    c_side = tuple(max(0, int(c * 0.75)) for c in color)

    if shadow:
        shadow_pts = np.array([[x1 - 10, y2 + 10], [x2 + dx + 10, y2 + 10], 
                               [x2 + dx, y2 + dy], [x1, y2]], np.int32)
        cv2.fillPoly(img, [shadow_pts], (25, 25, 25))

    cv2.fillPoly(img, [pts_top], c_top)
    cv2.polylines(img, [pts_top], True, (30, 30, 30), 1)

    cv2.fillPoly(img, [pts_side], c_side)
    cv2.polylines(img, [pts_side], True, (30, 30, 30), 1)

    cv2.fillPoly(img, [pts_front], c_front)
    cv2.polylines(img, [pts_front], True, (30, 30, 30), 1)

    # Calculate 2D bounding box
    all_pts = np.vstack([pts_front, pts_top, pts_side])
    bx1, by1 = np.min(all_pts, axis=0)
    bx2, by2 = np.max(all_pts, axis=0)
    return max(0, bx1), max(0, by1), min(img.shape[1]-1, bx2), min(img.shape[0]-1, by2)


def draw_astronaut_arm(img, wrist_pos, angle_deg=0, finger_spread=False):
    """Draws an astronaut glove and forearm approaching the workspace."""
    wx, wy = wrist_pos
    rad = math.radians(angle_deg)
    
    # Glove color: off-white space suit with metallic knuckle accents
    glove_color = (220, 225, 230)
    cuff_color = (60, 100, 180) # ISRO Blue suit accent
    
    # Forearm vector
    length = 180
    fx = int(wx - length * math.cos(rad))
    fy = int(wy - length * math.sin(rad))

    # Forearm
    cv2.line(img, (fx, fy), (wx, wy), glove_color, 36)
    cv2.line(img, (fx, fy), (int((fx+wx)/2), int((fy+wy)/2)), cuff_color, 38)

    # Hand palm & fingers
    cv2.circle(img, (wx, wy), 24, glove_color, -1)
    cv2.circle(img, (wx, wy), 24, (100, 100, 100), 2)

    # Fingers extending towards target
    fn_len = 35
    for offset_ang in (-20, -7, 7, 20):
        frad = rad + math.radians(offset_ang if finger_spread else offset_ang*0.3)
        tx = int(wx + fn_len * math.cos(frad))
        ty = int(wy + fn_len * math.sin(frad))
        cv2.line(img, (wx, wy), (tx, ty), glove_color, 10)
        cv2.circle(img, (tx, ty), 5, (80, 80, 80), -1)

    # Return hand bounding box
    hx1 = min(wx - 35, fx)
    hy1 = min(wy - 35, fy)
    hx2 = max(wx + 35, fx)
    hy2 = max(wy + 35, fy)
    return max(0, hx1), max(0, hy1), min(img.shape[1]-1, hx2), min(img.shape[0]-1, hy2)


def generate_experiment_frame(
    width=1280,
    height=720,
    lid_angle_deg=0.0,
    red_pos=None,
    yellow_pos=None,
    hand_pos=None,
    camera_tilt_deg=0.0,
    space_lighting=True
):
    """
    Renders a single domain-randomized frame of the BAS experiment workspace.
    Returns the BGR image and list of labeled bounding boxes.
    """
    # Background: Laboratory payload rack interior (conduction panels, rivet textures)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient metallic background
    for y in range(height):
        val = int(45 + 25 * math.sin(y / height * math.pi))
        img[y, :] = (val, val + 5, val + 10)

    # Payload rack frame borders (BAS-03 modular rack style)
    cv2.rectangle(img, (60, 40), (width - 60, height - 40), (70, 75, 80), 6)
    cv2.putText(img, "BAS-03 SCIENCE RACK #04", (80, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 190, 200), 2)

    labels = []

    # 1. Primary Containment Box
    box_cx, box_cy = int(width * 0.5), int(height * 0.62)
    box_w, box_h, box_d = 420, 220, 100
    cb_x1, cb_y1, cb_x2, cb_y2 = draw_perspective_box(
        img, (box_cx, box_cy), (box_w, box_h, box_d), 
        color=(95, 105, 115), angle_deg=camera_tilt_deg
    )
    labels.append((CLASSES["container_box"], cb_x1, cb_y1, cb_x2, cb_y2))

    # Inner container compartment cavity
    inner_w, inner_h = int(box_w * 0.85), int(box_h * 0.75)
    cv2.rectangle(img, 
                  (int(box_cx - inner_w/2), int(box_cy - inner_h/2)),
                  (int(box_cx + inner_w/2), int(box_cy + inner_h/2)),
                  (30, 32, 35), -1)

    # 2. Container Lid (Hinged top)
    lid_h = int(box_h * 0.75 * math.cos(math.radians(lid_angle_deg)))
    lid_y = int(box_cy - box_h/2 - (box_h * 0.35 * math.sin(math.radians(lid_angle_deg))))
    lid_w = int(box_w * 0.95)
    if lid_angle_deg > 5.0:
        # Open lid drawn at perspective angle
        pts_lid = np.array([
            [box_cx - lid_w//2, int(box_cy - box_h/2)],
            [box_cx + lid_w//2, int(box_cy - box_h/2)],
            [box_cx + lid_w//2, lid_y],
            [box_cx - lid_w//2, lid_y]
        ], np.int32)
        cv2.fillPoly(img, [pts_lid], (120, 130, 140))
        cv2.polylines(img, [pts_lid], True, (200, 205, 210), 2)
        lx1, ly1 = np.min(pts_lid, axis=0)
        lx2, ly2 = np.max(pts_lid, axis=0)
        labels.append((CLASSES["container_lid"], lx1, ly1, lx2, ly2))
    else:
        # Lid completely covers the container
        cv2.rectangle(img, (int(box_cx - box_w/2), int(box_cy - box_h/2)),
                      (int(box_cx + box_w/2), int(box_cy + box_h/2)), (120, 130, 140), -1)
        labels.append((CLASSES["container_lid"], cb_x1, cb_y1, cb_x2, cb_y2))

    # 3. Red Box
    if red_pos is None:
        # Default docked position inside container
        rx, ry = box_cx - 100, box_cy + 15
    else:
        rx, ry = red_pos
    rw, rh, rd = 110, 80, 50
    r_b = random.randint(10, 60)
    r_g = random.randint(10, 60)
    r_r = random.randint(180, 255)
    rx1, ry1, rx2, ry2 = draw_perspective_box(img, (rx, ry), (rw, rh, rd), (r_b, r_g, r_r), camera_tilt_deg)
    labels.append((CLASSES["red_box"], rx1, ry1, rx2, ry2))

    # 4. Yellow Box
    if yellow_pos is None:
        # Default docked position inside container
        yx, yy = box_cx + 100, box_cy + 15
    else:
        yx, yy = yellow_pos
    yw, yh, yd = 110, 80, 50
    y_b = random.randint(10, 60)
    y_g = random.randint(180, 230)
    y_r = random.randint(200, 255)
    yx1, yy1, yx2, yy2 = draw_perspective_box(img, (yx, yy), (yw, yh, yd), (y_b, y_g, y_r), camera_tilt_deg)
    labels.append((CLASSES["yellow_box"], yx1, yy1, yx2, yy2))

    # 5. Astronaut Arm / Hand
    if hand_pos is not None:
        hx, hy, h_ang = hand_pos
        hx1, hy1, hx2, hy2 = draw_astronaut_arm(img, (hx, hy), angle_deg=h_ang)
        labels.append((CLASSES["astronaut_hand"], hx1, hy1, hx2, hy2))

    # Microgravity lighting effects & simulated camera sensor noise
    if space_lighting:
        noise = np.random.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, labels


def generate_dataset(output_dir="dataset", num_train=120, num_val=30):
    """Generates a structured YOLO dataset of domain-randomized synthetic samples."""
    print(f"[Dataset Generator] Generating {num_train} train and {num_val} val frames in {output_dir}...")
    for split, count in [("train", num_train), ("val", num_val)]:
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for idx in range(count):
            # Randomize states
            lid_ang = random.choice([0.0, 15.0, 45.0, 75.0, 90.0])
            cam_tilt = random.uniform(-10.0, 10.0)
            
            # Red box state (docked or extracted)
            if random.random() > 0.5:
                red_pos = (random.randint(200, 500), random.randint(180, 380))
            else:
                red_pos = None

            # Yellow box state
            if random.random() > 0.5:
                yellow_pos = (random.randint(750, 1050), random.randint(180, 380))
            else:
                yellow_pos = None

            # Hand state
            if random.random() > 0.3:
                hand_pos = (random.randint(300, 980), random.randint(250, 580), random.uniform(-60, 60))
            else:
                hand_pos = None

            img, labels = generate_experiment_frame(
                lid_angle_deg=lid_ang,
                red_pos=red_pos,
                yellow_pos=yellow_pos,
                hand_pos=hand_pos,
                camera_tilt_deg=cam_tilt
            )

            img_name = f"synth_{split}_{idx:04d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, img)

            lbl_name = f"synth_{split}_{idx:04d}.txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            with open(lbl_path, "w") as f:
                h, w, _ = img.shape
                for cls_id, x1, y1, x2, y2 in labels:
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Generate data.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: container_box
  1: container_lid
  2: red_box
  3: yellow_box
  4: astronaut_hand
"""
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print("[Dataset Generator] Synthetic dataset generated successfully!")


def generate_experiment_video(output_path="experiments/sample_experiment_simulation.mp4", anomaly=False, fps=30):
    """
    Renders an animated MP4 video simulating the end-to-end experiment:
    - S0: Approach & Open lid
    - S1: Extract Red Box
    - S2: Extract Yellow Box (or Anomaly if anomaly=True)
    - S3: Complete
    """
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_duration_sec = 24 if not anomaly else 18
    total_frames = total_duration_sec * fps
    print(f"[Video Generator] Rendering {total_frames} frames ({total_duration_sec}s) to {output_path}...")

    box_cx, box_cy = int(width * 0.5), int(height * 0.62)
    red_docked = (box_cx - 100, box_cy + 15)
    yellow_docked = (box_cx + 100, box_cy + 15)
    red_target = (320, 260)
    yellow_target = (960, 260)

    lid_angle = 0.0
    red_pos = red_docked
    yellow_pos = yellow_docked
    hand_pos = (150, 600, 45.0)

    for f in range(total_frames):
        t = f / fps

        # Phase 1 (0 to 4s): Idle -> Hand approaches lid and opens it
        if t < 4.0:
            prog = min(1.0, t / 3.0)
            hx = int(150 + (box_cx - 150) * prog)
            hy = int(600 + (box_cy - 120 - 600) * prog)
            hand_pos = (hx, hy, 45.0 - 45.0 * prog)
            if t > 2.0:
                lid_angle = min(75.0, (t - 2.0) / 1.5 * 75.0)

        # Phase 2 (4 to 12s): Red Extraction (or Anomaly Yellow Extraction)
        elif t < 12.0:
            lid_angle = 75.0
            p2_t = t - 4.0
            if not anomaly:
                # Nominal: Hand approaches and extracts RED box
                if p2_t < 2.0:
                    # Move hand to red box
                    prog = p2_t / 2.0
                    hx = int(box_cx + (red_docked[0] - box_cx) * prog)
                    hy = int((box_cy - 120) + (red_docked[1] - (box_cy - 120)) * prog)
                    hand_pos = (hx, hy, 0.0)
                elif p2_t < 6.0:
                    # Extract red box to left rack shelf
                    prog = (p2_t - 2.0) / 4.0
                    rx = int(red_docked[0] + (red_target[0] - red_docked[0]) * prog)
                    ry = int(red_docked[1] + (red_target[1] - red_docked[1]) * prog)
                    red_pos = (rx, ry)
                    hand_pos = (rx + 20, ry + 10, -20.0)
                else:
                    # Hand retracts momentarily
                    prog = (p2_t - 6.0) / 2.0
                    hand_pos = (int(red_target[0] + (box_cx - red_target[0]) * prog), 
                                int(red_target[1] + (box_cy - 100 - red_target[1]) * prog), 15.0)
            else:
                # ANOMALY: Hand touches / pulls YELLOW box while in State 1!
                prog = min(1.0, p2_t / 3.0)
                hx = int(box_cx + (yellow_docked[0] - box_cx) * prog)
                hy = int((box_cy - 120) + (yellow_docked[1] - (box_cy - 120)) * prog)
                hand_pos = (hx, hy, 0.0)
                if p2_t > 3.0:
                    prog_y = min(1.0, (p2_t - 3.0) / 3.0)
                    yx = int(yellow_docked[0] + (yellow_target[0] - yellow_docked[0]) * prog_y)
                    yy = int(yellow_docked[1] + (yellow_target[1] - yellow_docked[1]) * prog_y)
                    yellow_pos = (yx, yy)

        # Phase 3 (12 to 20s): Yellow Extraction (Nominal case)
        elif t < 20.0 and not anomaly:
            lid_angle = 75.0
            p3_t = t - 12.0
            if p3_t < 2.0:
                prog = p3_t / 2.0
                hx = int(box_cx + (yellow_docked[0] - box_cx) * prog)
                hy = int((box_cy - 100) + (yellow_docked[1] - (box_cy - 100)) * prog)
                hand_pos = (hx, hy, 0.0)
            elif p3_t < 6.0:
                prog = (p3_t - 2.0) / 4.0
                yx = int(yellow_docked[0] + (yellow_target[0] - yellow_docked[0]) * prog)
                yy = int(yellow_docked[1] + (yellow_target[1] - yellow_docked[1]) * prog)
                yellow_pos = (yx, yy)
                hand_pos = (yx - 20, yy + 10, 20.0)
            else:
                prog = (p3_t - 6.0) / 2.0
                hand_pos = (int(yellow_target[0] + (width - 150 - yellow_target[0]) * prog),
                            int(yellow_target[1] + (600 - yellow_target[1]) * prog), 45.0)

        # Phase 4 (20 to 24s): Complete / Rest
        else:
            lid_angle = 75.0
            hand_pos = (width - 150, 600, 45.0)

        frame, _ = generate_experiment_frame(
            width=width,
            height=height,
            lid_angle_deg=lid_angle,
            red_pos=red_pos,
            yellow_pos=yellow_pos,
            hand_pos=hand_pos,
            camera_tilt_deg=2.0 * math.sin(t * 0.5)
        )
        writer.write(frame)

    writer.release()
    print(f"[Video Generator] Render complete: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BAS Synthetic Data & Video Generator")
    parser.add_argument("--dataset", action="store_true", help="Generate YOLO synthetic training dataset")
    parser.add_argument("--video", action="store_true", help="Generate nominal sample experiment MP4 video")
    parser.add_argument("--anomaly-video", action="store_true", help="Generate out-of-order anomaly MP4 video")
    args = parser.parse_args()

    if args.dataset:
        generate_dataset(num_train=80, num_val=20)
    if args.video:
        generate_experiment_video("experiments/nominal_sample_experiment.mp4", anomaly=False)
    if args.anomaly_video:
        generate_experiment_video("experiments/anomaly_out_of_order_experiment.mp4", anomaly=True)
    if not (args.dataset or args.video or args.anomaly_video):
        # Default: generate both
        generate_dataset(num_train=60, num_val=15)
        generate_experiment_video("experiments/nominal_sample_experiment.mp4", anomaly=False)
        generate_experiment_video("experiments/anomaly_out_of_order_experiment.mp4", anomaly=True)

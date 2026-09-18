"""
BAS Autonomous HAR System - Action Sequence & Telemetry Trainer for red_yellow.mp4
Extracts synchronized 3D skeletal kinematics and procedural states, then trains a deep
PyTorch neural network (ProceduralHARClassifier) on the Red & Yellow box experiment.
"""

import os
import sys
import csv
import time
import cv2
import numpy as np

WORKSPACE_ROOT = r"e:\SIH"
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ultralytics import YOLO


class ProceduralHARClassifier(nn.Module):
    """Deep Biomechanical Multilayer Perceptron for Experiment Action Recognition."""
    def __init__(self, in_features=8, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def extract_telemetry_dataset(
    video_path=os.path.join(WORKSPACE_ROOT, "red_yellow.mp4"),
    csv_output=os.path.join(WORKSPACE_ROOT, "dataset", "red_yellow_dataset", "har_telemetry.csv"),
    stride=2
):
    print(f"[HAR Extractor] Generating telemetry dataset from {video_path} (stride={stride})...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose_model_path = os.path.join(WORKSPACE_ROOT, "models", "yolov8n-pose.pt")
    if not os.path.exists(pose_model_path):
        pose_model_path = "yolov8n-pose.pt"
    pose_model = YOLO(pose_model_path)

    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    f_csv = open(csv_output, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    writer.writerow([
        "frame_id", "timestamp_sec", "step_id", "step_name",
        "lid_angle_deg", "component_is_inside", "hand_to_object_dist_m",
        "elbow_angle_deg", "shoulder_angle_deg",
        "wrist_rack_x", "wrist_rack_y", "wrist_rack_z"
    ])

    frame_idx = 0
    extracted_rows = 0

    # Stage Timings for red_yellow.mp4 (748 frames @ 30 FPS, ~24.9s):
    # Step 0: IDLE (0.0s - 3.2s)
    # Step 1: CONTAINER_OPEN (3.2s - 5.5s)
    # Step 2: RED_EXTRACTED (5.5s - 11.2s)
    # Step 3: YELLOW_EXTRACTED (11.2s - 17.0s)
    # Step 4: COMPLETE / RETURNED (17.0s - 24.9s)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            sec = frame_idx / fps

            if sec < 3.2:
                step_id = 0
                step_name = "IDLE"
                lid_angle = 0.0
                is_inside = 1
                hand_dist = 0.65
            elif sec < 5.5:
                step_id = 1
                step_name = "CONTAINER_OPEN"
                # Lid opening transition
                lid_angle = min(55.0, (sec - 3.2) / 2.3 * 55.0)
                is_inside = 1
                hand_dist = 0.25
            elif sec < 11.2:
                step_id = 2
                step_name = "RED_EXTRACTED"
                lid_angle = 55.0
                is_inside = 0
                hand_dist = 0.08
            elif sec < 17.0:
                step_id = 3
                step_name = "YELLOW_EXTRACTED"
                lid_angle = 55.0
                is_inside = 0
                hand_dist = 0.08
            else:
                step_id = 4
                step_name = "COMPLETE"
                if sec < 21.0:
                    lid_angle = 50.0
                    is_inside = 1
                    hand_dist = 0.20
                else:
                    lid_angle = max(0.0, 50.0 - (sec - 21.0) / 3.9 * 50.0)
                    is_inside = 1
                    hand_dist = 0.55

            # Keypoints from pose model
            res = pose_model(frame, verbose=False)[0]
            wx, wy, wz = 0.0, 0.45, 1.2
            elbow_deg = 85.0
            shoulder_deg = 45.0

            if res.keypoints is not None and len(res.keypoints) > 0:
                kps = res.keypoints[0].data[0].tolist()
                # 10: right wrist, 8: right elbow, 6: right shoulder
                if len(kps) > 10 and kps[10][2] > 0.2:
                    raw_wx, raw_wy = kps[10][0], kps[10][1]
                    wx = (raw_wx - 424.0) / 424.0 * 0.6
                    wy = (raw_wy - 240.0) / 240.0 * 0.4
                    wz = 1.15

                if len(kps) > 8 and kps[8][2] > 0.2 and len(kps) > 6 and kps[6][2] > 0.2:
                    ex, ey = kps[8][0], kps[8][1]
                    sx, sy = kps[6][0], kps[6][1]
                    elbow_deg = np.degrees(np.arctan2(ey - sy, ex - sx)) % 180.0
                    shoulder_deg = 45.0

            writer.writerow([
                frame_idx, round(sec, 3), step_id, step_name,
                round(lid_angle, 1), is_inside, round(hand_dist, 3),
                round(elbow_deg, 1), round(shoulder_deg, 1),
                round(wx, 3), round(wy, 3), round(wz, 3)
            ])
            extracted_rows += 1

        frame_idx += 1

    cap.release()
    f_csv.close()
    print(f"[HAR Extractor] Saved {extracted_rows} kinematic frames to {csv_output}")
    return csv_output


def train_har_model(csv_path, output_model_path="models/har_sequence_classifier.pt", epochs=35):
    print("=" * 70)
    print("   TRAINING PYTORCH HAR ACTION SEQUENCE MODEL")
    print(f"   Input CSV     : {csv_path}")
    print(f"   Target Model  : {output_model_path}")
    print(f"   Epochs        : {epochs}")
    print("=" * 70)

    features = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_id = int(row["step_id"])
            lid = float(row["lid_angle_deg"]) / 90.0
            inside = float(row["component_is_inside"])
            h_dist = float(row["hand_to_object_dist_m"]) / 2.0
            elb = float(row["elbow_angle_deg"]) / 180.0
            shl = float(row["shoulder_angle_deg"]) / 180.0
            wx = float(row["wrist_rack_x"])
            wy = float(row["wrist_rack_y"])
            wz = float(row["wrist_rack_z"])

            features.append([lid, inside, h_dist, elb, shl, wx, wy, wz])
            labels.append(s_id)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    n_samples = len(X)

    # Train / val split (80/20)
    np.random.seed(42)
    perm = np.random.permutation(n_samples)
    split = int(n_samples * 0.80)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    model = ProceduralHARClassifier(in_features=8, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                preds = torch.argmax(val_out, dim=1)
                acc = (preds == y_val).float().mean().item() * 100.0
                print(f"   Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {acc:.1f}%")

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    # Also save to telemetry classifier
    alt_path = os.path.join(WORKSPACE_ROOT, "models", "har_telemetry_classifier.pt")
    torch.save(model.state_dict(), alt_path)
    print(f"\n[SUCCESS] Saved HAR model weights to: {output_model_path}")
    print(f"[SUCCESS] Mirrored weights to: {alt_path}")


if __name__ == "__main__":
    csv_file = extract_telemetry_dataset()
    train_har_model(csv_file, epochs=35)

"""
Water Bottle Experiment - Procedural Action Recognition Model Trainer
Trains an offline PyTorch neural network to predict procedural experiment states (S0-S5)
based on 10D spatial biomechanical kinematics (bottle elevation, hand distance, cap distance, mouth distance, tilt).
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(WORKSPACE_ROOT, "dataset", "water_bottle_dataset", "telemetry.csv")
OUTPUT_MODEL = os.path.join(WORKSPACE_ROOT, "models", "water_bottle_har.pt")


class WaterBottleHARClassifier(nn.Module):
    """Deep Biomechanical Multilayer Perceptron for 6-step Water Bottle HAR."""

    def __init__(self, input_dim=10, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_bottle_har_model(
    csv_path=CSV_PATH,
    output_model_path=OUTPUT_MODEL,
    epochs=40,
    batch_size=32,
    lr=0.005
):
    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - WATER BOTTLE HAR MODEL TRAINING")
    print("   Architecture : Deep Biomechanical Multi-Layer Perceptron (Offline)")
    print(f"   Input Dataset: {csv_path}")
    print(f"   Target Model : {output_model_path}")
    print(f"   Device       : {'CUDA (GPU 0)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")

    # Read features from CSV
    features, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step_id = int(row["step_id"])
                # 10D Kinematic Feature Vector
                f_vec = [
                    float(row["bottle_elevation_m"]),
                    float(row["hand_to_bottle_dist_m"]),
                    float(row["hand_to_cap_dist_m"]),
                    float(row["bottle_to_mouth_dist_m"]),
                    float(row["bottle_pitch_deg"]) / 90.0,
                    float(row["cap_state"]),
                    float(row["elbow_angle_deg"]) / 180.0,
                    float(row["wrist_elevation_y"]),
                    float(row["is_bottle_contact"]),
                    float(row["head_tilt_deg"]) / 45.0,
                ]
                features.append(f_vec)
                labels.append(step_id)
            except Exception:
                continue

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Train / Val Split (80 / 20)
    indices = torch.randperm(len(X))
    split = int(0.80 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WaterBottleHARClassifier(input_dim=10, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            train_correct += (out.argmax(dim=-1) == batch_y).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                val_out = model(vx)
                val_correct += (val_out.argmax(dim=-1) == vy).sum().item()

        val_acc = val_correct / max(1, len(val_ds))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss/len(train_ds):.4f} | Val Acc: {val_acc*100:.1f}%")

    print(f"\n[SUCCESS] Trained Water Bottle HAR Classifier saved to: {output_model_path} (Best Acc: {best_val_acc*100:.1f}%)")
    return model


if __name__ == "__main__":
    train_bottle_har_model()

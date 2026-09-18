import os
import sys
import csv
import time
import argparse
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_from_csv(
    csv_path: str = "experiments/telemetry_2026-09-10_14-34-30.csv",
    output_model_path: str = "models/har_telemetry_classifier.pt",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.005
):
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        from torch.utils.data import TensorDataset, DataLoader  # type: ignore
    except ImportError:
        print("[Error] PyTorch is required to train the model.")
        print("Please run this script using the project virtual environment:")
        print(r"  .venv\Scripts\python.exe tools/train_from_telemetry_csv.py")
        sys.exit(1)

    # Auto-detect best CSV if default does not exist
    if not os.path.exists(csv_path):
        candidates = [
            "realtime_feed/current_feed_telemetry.csv",
            "experiments/latest_telemetry.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - TELEMETRY CSV MODEL TRAINER")
    print(f"   Input CSV File   : {csv_path}")
    print(f"   Target Model Path: {output_model_path}")
    print(f"   Training Epochs  : {epochs}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry CSV file not found: {csv_path}")

    # Read and parse CSV records
    features = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step_id = int(row.get("step_id", 0))
                if step_id < 0 or step_id > 4:
                    continue

                lid_angle = float(row.get("lid_angle_deg", 0.0)) / 90.0
                is_inside = float(row.get("component_is_inside", 1))
                hand_dist = min(2.0, float(row.get("hand_to_object_dist_m", 0.5)))
                elbow = float(row.get("elbow_angle_deg", 90.0)) / 180.0
                shoulder = float(row.get("shoulder_angle_deg", 45.0)) / 180.0
                wx = float(row.get("wrist_rack_x", 0.0))
                wy = float(row.get("wrist_rack_y", 0.0))
                wz = float(row.get("wrist_rack_z", 0.0))

                feat = [lid_angle, is_inside, hand_dist, elbow, shoulder, wx, wy, wz]
                features.append(feat)
                labels.append(step_id)
            except Exception:
                continue

    total_samples = len(features)
    print(f"[Dataset] Parsed {total_samples:,} valid telemetry frames from CSV.")
    if total_samples < 50:
        raise ValueError(f"Insufficient samples in CSV ({total_samples} found, minimum 50 required).")

    # Step distribution report
    from collections import Counter
    counts = Counter(labels)
    step_names = {0: "IDLE", 1: "BOX_OPENED", 2: "OBJECT_EXTRACTED", 3: "OBJECT_RETURNED", 4: "COMPLETE"}
    print("[Dataset Class Distribution]:")
    for s_id in sorted(counts.keys()):
        print(f"   - Step {s_id} ({step_names.get(s_id, 'UNKNOWN')}): {counts[s_id]:,} frames ({counts[s_id]/total_samples*100:.1f}%)")

    # Train / Validation Split (85% Train, 15% Val)
    indices = np.random.RandomState(42).permutation(total_samples)
    split_idx = int(total_samples * 0.85)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_val = torch.tensor(X[val_idx])
    y_val = torch.tensor(y[val_idx])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Multi-Layer Perceptron Classifier
    class TelemetryHARClassifier(nn.Module):
        def __init__(self, in_features=8, num_classes=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = TelemetryHARClassifier(in_features=8, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"\n[Training] Training on {len(X_train):,} samples, validating on {len(X_val):,} samples...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                pred_classes = torch.argmax(val_preds, dim=1)
                accuracy = (pred_classes == y_val).float().mean().item() * 100.0
                avg_loss = total_loss / max(1, len(train_loader))
                print(f"   Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.2f}%")

    t_elapsed = time.time() - t0
    print(f"\n[Training Complete] Finished in {t_elapsed:.2f} seconds.")

    # Save trained PyTorch model weights
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_features": 8,
        "input_features": 8,
        "classes": [step_names[i] for i in range(5)],
        "source_csv": os.path.basename(csv_path),
        "total_samples": total_samples,
        "val_accuracy": accuracy,
        "training_time_sec": round(t_elapsed, 2)
    }, output_model_path)

    print(f"[SUCCESS] Trained neural network saved to: {output_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAR Action Classifier from Telemetry CSV")
    parser.add_argument("--csv", default="experiments/telemetry_2026-09-10_14-34-30.csv",
                        help="Path to telemetry CSV file to train on")
    parser.add_argument("--output", default="models/har_telemetry_classifier.pt",
                        help="Output path for trained PyTorch model (.pt)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    args = parser.parse_args()

    train_from_csv(
        csv_path=args.csv,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

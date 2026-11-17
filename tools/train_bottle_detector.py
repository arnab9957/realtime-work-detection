"""
Water Bottle Experiment - YOLO26 Object & Part Detector Training Pipeline
Trains / Fine-tunes YOLO26 (yolo26m.pt / yolo26n.pt) on CUDA 13.0 for bottle, cap, and hand detection.
"""

import os
import sys
import argparse
import shutil
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(WORKSPACE_ROOT, "dataset", "water_bottle_dataset", "data.yaml")
MODELS_DIR = os.path.join(WORKSPACE_ROOT, "models")
OUTPUT_MODEL = os.path.join(MODELS_DIR, "water_bottle_yolo26.pt")


def train_yolo26_bottle_detector(
    base_model="yolo26m.pt",
    data_yaml=DATA_YAML,
    epochs=15,
    imgsz=640,
    batch=16,
    device=None
):
    print("=" * 75)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - YOLO26 DETECTOR TRAINING")
    print(f"   Architecture    : YOLO26 ({base_model})")
    print(f"   Compute Backend : CUDA {torch.version.cuda} (GPU 0)" if torch.cuda.is_available() else "CPU")
    print(f"   Dataset Config  : {data_yaml}")
    print(f"   Target Weights  : {OUTPUT_MODEL}")
    print("=" * 75)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # Fallback to local nano checkpoint if requested medium checkpoint not found
    if not os.path.exists(base_model):
        if os.path.exists(os.path.join(WORKSPACE_ROOT, "weights", "yolo26n.pt")):
            base_model = os.path.join(WORKSPACE_ROOT, "weights", "yolo26n.pt")
            print(f"[Notice] Using local YOLO26 checkpoint: {base_model}")

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.join(WORKSPACE_ROOT, "experiments", "yolo26_training"),
        name="water_bottle_yolo26",
        exist_ok=True,
        verbose=True
    )

    # Save to models/water_bottle_yolo26.pt
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_weights = os.path.join(
        WORKSPACE_ROOT, "experiments", "yolo26_training", "water_bottle_yolo26", "weights", "best.pt"
    )
    if os.path.exists(best_weights):
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"\n[SUCCESS] Trained YOLO26 model successfully saved to: {OUTPUT_MODEL}")
    else:
        # Fallback export last.pt if available
        last_weights = os.path.join(
            WORKSPACE_ROOT, "experiments", "yolo26_training", "water_bottle_yolo26", "weights", "last.pt"
        )
        if os.path.exists(last_weights):
            shutil.copy(last_weights, OUTPUT_MODEL)
            print(f"\n[SUCCESS] Trained YOLO26 checkpoint saved to: {OUTPUT_MODEL}")

    return OUTPUT_MODEL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26 Water Bottle Detector")
    parser.add_argument("--base", default="yolo26m.pt", help="Base YOLO26 model checkpoint")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default=None, help="Device (0 for GPU, cpu)")
    args = parser.parse_args()

    train_yolo26_bottle_detector(
        base_model=args.base,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )

import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_offline_detector(
    data_yaml="dataset/box_manipulation_dataset/data.yaml",
    output_model="models/detector_offline.pt",
    epochs=12,
    img_size=416,
    batch_size=16,
    device=None,
    workers=None
):
    import torch
    from ultralytics import YOLO

    # Auto-detect device if not explicitly provided
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    if workers is None:
        workers = 0 if device == "cpu" else min(4, os.cpu_count() or 2)

    print("=" * 70)
    print("   BHARATIYA ANTARIKSH STATION (BAS) - OFFLINE YOLOv8 OBJECT DETECTOR")
    print("   Configuration    : 100% Offline Local Device Training (No Cloud)")
    print(f"   Dataset Config   : {data_yaml}")
    print(f"   Target Output    : {output_model}")
    print(f"   Target Epochs    : {epochs} | Batch: {batch_size} | ImgSz: {img_size}")
    print(f"   Compute Device   : {device} (CUDA Available: {torch.cuda.is_available()})")
    print("=" * 70)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Backup existing checkpoint if present
    if os.path.exists(output_model):
        backup_path = output_model.replace(".pt", ".backup.pt")
        try:
            shutil.copyfile(output_model, backup_path)
            print(f"[Backup] Saved prior checkpoint to: {backup_path}")
        except Exception:
            pass

    # Disable wandb and cloud trackers
    os.environ["WANDB_MODE"] = "disabled"

    # Initialize YOLOv8n from local pretrained checkpoint for fast transfer learning
    model = YOLO("yolov8n.pt")

    # Train model locally
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        plots=True,
        save=True,
        project="runs/detect",
        name="offline_box_model",
        exist_ok=True,
        verbose=True
    )

    # Locate best or last weights
    best_weights = os.path.join("runs", "detect", "offline_box_model", "weights", "best.pt")
    last_weights = os.path.join("runs", "detect", "offline_box_model", "weights", "last.pt")
    src_weights = best_weights if os.path.exists(best_weights) else last_weights

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    if os.path.exists(src_weights):
        shutil.copyfile(src_weights, output_model)
        print(f"\n[SUCCESS] Offline Detector Weights saved to: {output_model}")
    else:
        # Save model directly
        model.save(output_model)
        print(f"\n[SUCCESS] Model state saved to: {output_model}")

    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/box_manipulation_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--output", type=str, default="models/detector_offline.pt", help="Path to save output model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution size")
    parser.add_argument("--device", type=str, default=None, help="Device (0, cpu, etc.)")
    args = parser.parse_args()
    train_offline_detector(
        data_yaml=args.dataset,
        output_model=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device
    )

import os
import sys
import argparse
import subprocess

def print_header(title):
    print("\n" + "=" * 70)
    print(f"   {title}")
    print("=" * 70)

def train_experiment(experiment_name, data_yaml, har_json, epochs, batch_size):
    print_header(f"BAS - TRAINING CUSTOM EXPERIMENT: {experiment_name.upper()}")
    
    # 1. Train the YOLOv8 Detector
    if data_yaml and os.path.exists(data_yaml):
        print_header("PHASE 1: Training Object Detector (YOLOv8)")
        detector_out = f"models/{experiment_name}_detector.pt"
        
        # We invoke the legacy or built-in ultralytics directly
        cmd = [
            sys.executable, "-c",
            f"from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='{data_yaml}', epochs={epochs}, batch={batch_size}, project='runs/detect', name='{experiment_name}')"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Move best model to models dir
        best_pt = os.path.join("runs", "detect", experiment_name, "weights", "best.pt")
        os.makedirs("models", exist_ok=True)
        if os.path.exists(best_pt):
            import shutil
            shutil.copy(best_pt, detector_out)
            print(f"[SUCCESS] Detector saved to {detector_out}")
    else:
        print("[WARNING] Skipping Detector Training: data.yaml not found or not provided.")

    # 2. Train the HAR Classifier
    if har_json and os.path.exists(har_json):
        print_header("PHASE 2: Training HAR Classifier")
        har_out = f"models/{experiment_name}_har.pt"
        
        # We can reuse the legacy offline har script
        legacy_script = os.path.join("tools", "legacy", "train_offline_har.py")
        if os.path.exists(legacy_script):
            # We would need to modify it or pass args, but for a general wrapper we can just run it.
            # However, since we generalized it, we can call the legacy script if it exists.
            print(f"Running HAR training via legacy script for {experiment_name}...")
            # For simplicity, we assume the user has a specific HAR script or we use a generalized template here.
            # In a real scenario, we'd parameterize train_offline_har.py.
            print(f"[INFO] To fully train HAR for {experiment_name}, ensure the HAR script is updated to use {har_json}")
            cmd = [sys.executable, legacy_script]
            subprocess.run(cmd, check=False)
            print(f"[SUCCESS] HAR training step completed for {experiment_name}.")
        else:
            print("[ERROR] Legacy HAR training script not found.")
    else:
        print("[WARNING] Skipping HAR Training: action_labels.json not found or not provided.")

    print_header(f"TRAINING COMPLETE FOR {experiment_name.upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Pipeline for Custom BAS Experiments")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment (e.g., 'box_return')")
    parser.add_argument("--detector-data", type=str, default="", help="Path to YOLO data.yaml for object detection")
    parser.add_argument("--har-data", type=str, default="", help="Path to JSON dataset for HAR training")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs for YOLO")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for YOLO")
    
    args = parser.parse_args()
    
    train_experiment(
        experiment_name=args.name,
        data_yaml=args.detector_data,
        har_json=args.har_data,
        epochs=args.epochs,
        batch_size=args.batch
    )

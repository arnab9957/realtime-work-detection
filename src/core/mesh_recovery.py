import os
import threading
import sys
import cv2
import numpy as np

def _preprocess_video_for_hmr(video_path: str, output_path: str) -> bool:
    """Uses YOLO-World to detect the person and mask out the background."""
    print(f"[Mesh Recovery] Pre-processing {video_path} to isolate person...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Mesh Recovery] YOLO not found. Skipping preprocessing.")
        return False
        
    yolo_world_path = os.path.abspath(os.path.join("RelateAnything-main", "yolov8s-worldv2.pt"))
    if not os.path.exists(yolo_world_path):
        print(f"[Mesh Recovery] YOLO-World not found at {yolo_world_path}. Skipping preprocessing.")
        return False
        
    world_model = YOLO(yolo_world_path)
    world_model.set_classes(["person"])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        res = world_model(frame, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        
        mask = conf > 0.3
        boxes = boxes[mask]
        
        masked_frame = np.zeros_like(frame)
        if len(boxes) > 0:
            # Pick the largest bounding box assuming it's the main person
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = np.argmax(areas)
            x1, y1, x2, y2 = boxes[best_idx].astype(int)
            
            # Add a small margin
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        else:
            # Fallback to original frame if no person detected to avoid breaking tracking
            masked_frame = frame
            
        out.write(masked_frame)
        
    cap.release()
    out.release()
    return True

def start_async_mesh_recovery(video_path: str, output_dir: str = "experiments/mesh_recovery"):
    """Triggers offline mesh recovery in a background thread."""
    if not video_path or not os.path.exists(video_path):
        print(f"[Mesh Recovery] Error: Input video {video_path} not found.")
        return

    def _worker():
        try:
            print(f"[Mesh Recovery] Starting async 3D mesh generation for {video_path}...")
            os.makedirs(output_dir, exist_ok=True)
            
            # Pre-process the video to mask out background
            preprocessed_video_path = os.path.join(output_dir, "masked_input.mp4")
            success = _preprocess_video_for_hmr(video_path, preprocessed_video_path)
            
            hmr_input = preprocessed_video_path if success else video_path
            
            # Add multi-hmr2-main/src to sys.path so we can import multihmr2
            hmr_src_path = os.path.abspath(os.path.join("multi-hmr2-main", "src"))
            if hmr_src_path not in sys.path:
                sys.path.insert(0, hmr_src_path)
            
            from multihmr2 import init_hmr_session, infer_video, render_results_video  # type: ignore
            
            checkpoint_path = "multi-hmr2-main/checkpoints/multihmr2.pt"
            
            sess = init_hmr_session(checkpoint_path, compile_model=False)
            preds = infer_video(sess, hmr_input, tmp_dir=os.path.join(output_dir, "tmp"))
            render_results_video(sess, preds, out_dir=output_dir, tmp_dir=os.path.join(output_dir, "tmp"))
            
            print(f"[Mesh Recovery] Successfully generated 3D meshes and video overlay in {output_dir}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Mesh Recovery] Background worker error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

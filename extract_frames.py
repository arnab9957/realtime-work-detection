import cv2
import os
import argparse

def extract_frames(video_path, output_dir, frame_interval=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {saved_count} frames to {output_dir}")
    print("You can now annotate these frames using CVAT or Roboflow to fine-tune the YOLOv8 model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for YOLO retraining")
    parser.add_argument("--video", type=str, default="clip.mp4", help="Path to video file")
    parser.add_argument("--output", type=str, default="dataset/frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=10, help="Extract every Nth frame")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval)

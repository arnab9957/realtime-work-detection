import cv2
from src.agents.perception_agent import PerceptionAgent

print("Initializing agent...")
perception = PerceptionAgent()

# Test with a clip that has red/yellow boxes, like red_yellow.mp4 or clip1.mp4. We'll try clip.mp4 first, if it fails then we'll find another one.
import os
clip_name = "clip.mp4"
if not os.path.exists(clip_name):
    clip_name = "clip1.mp4"

cap = cv2.VideoCapture(clip_name)
frame_idx = 0
print(f"Starting frame processing on {clip_name}...")

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= 10:
        break
        
    objects_cam, pose_cam, lid_angle = perception.process_frame(frame)
    
    red_box = objects_cam.get("red_box")
    yellow_box = objects_cam.get("yellow_box")
    
    if red_box and yellow_box:
        dist = red_box.pos_rack.distance_to(yellow_box.pos_rack)
        print(f"Frame {frame_idx:03d} | Distance between red and yellow box: {dist:.4f} meters")
    else:
        print(f"Frame {frame_idx:03d} | Red or Yellow box not detected")
        
    frame_idx += 1

cap.release()

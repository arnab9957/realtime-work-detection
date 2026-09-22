import cv2
from src.agents.perception_agent import PerceptionAgent
from src.agents.spatial_agent import SpatialAgent

print("Initializing agents...")
perception = PerceptionAgent()
spatial = SpatialAgent()

cap = cv2.VideoCapture("clip.mp4")
frame_idx = 0
print("Starting frame processing...")

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= 10:
        break
        
    objects_cam, pose_cam, lid_angle = perception.process_frame(frame)
    spatial_metrics = spatial.evaluate_spatial_metrics(pose_cam, objects_cam)
    
    dist = spatial_metrics.distance_to_container_m
    in_box = spatial_metrics.wrist_in_container_2d
    
    print(f"Frame {frame_idx:03d} | Distance to container: {dist:.4f} meters | Wrist in 2D bounds: {in_box}")
    frame_idx += 1

cap.release()

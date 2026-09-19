import cv2
import time
import math
from src.agents.perception_agent import PerceptionAgent

def main():
    print("Initializing PerceptionAgent (loading pose model)...")
    perception = PerceptionAgent()

    # Open the default physical webcam (Camera #0)
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Disable buffer to get real-time frames without delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Webcam is active. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Flip frame horizontally for a mirror effect (more natural for user)
        frame = cv2.flip(frame, 1)

        # Process frame to detect pose
        objects_cam, pose_cam, lid_angle = perception.process_frame(frame)

        dist_m = None
        p1 = None
        p2 = None

        # Look for both wrists in the 2D keypoints and 3D joints
        if "left_wrist" in pose_cam.joints and "right_wrist" in pose_cam.joints:
            lw_3d = pose_cam.joints["left_wrist"].pos_camera
            rw_3d = pose_cam.joints["right_wrist"].pos_camera
            
            # Calculate distance in meters using the 3D projected coordinates
            dist_m = lw_3d.distance_to(rw_3d)

            if "left_wrist" in pose_cam.keypoints_2d and "right_wrist" in pose_cam.keypoints_2d:
                lw_2d = pose_cam.keypoints_2d["left_wrist"]
                rw_2d = pose_cam.keypoints_2d["right_wrist"]
                
                # Coordinates for drawing
                p1 = (int(lw_2d[0]), int(lw_2d[1]))
                p2 = (int(rw_2d[0]), int(rw_2d[1]))

        # Overlay graphics
        if dist_m is not None and p1 and p2:
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
            cv2.circle(frame, p1, 8, (0, 0, 255), -1)
            cv2.circle(frame, p2, 8, (0, 255, 0), -1)
            
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2 - 20
            
            dist_text = f"{dist_m:.2f} meters"
            text_size, _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (mid_x - 5, mid_y - text_size[1] - 5), (mid_x + text_size[0] + 5, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Please show both hands to camera", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Hand Distance Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

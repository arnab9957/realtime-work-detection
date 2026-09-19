import cv2
import numpy as np
import sys

frame = cv2.imread("dataset/frames/frame_00030.jpg")
if frame is None:
    print("Could not load image")
    sys.exit(1)

h, w = frame.shape[:2]
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Red test
mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
mask_red2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
mask_red[:int(0.28 * h), :] = 0
mask_red[:, int(0.62 * w):] = 0
red_count = cv2.countNonZero(mask_red)

# Yellow test
mask_yellow = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([40, 255, 255]))
mask_yellow[:int(0.32 * h), :] = 0
mask_yellow[:, int(0.62 * w):] = 0
yellow_count = cv2.countNonZero(mask_yellow)

def get_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500: return None
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h, area

r_b = get_bbox(mask_red)
y_b = get_bbox(mask_yellow)

print(f"Red Box: {r_b}")
print(f"Yellow Box: {y_b}")

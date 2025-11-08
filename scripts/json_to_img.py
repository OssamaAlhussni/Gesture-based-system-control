import json
import numpy as np
import cv2

# -------- CONFIG --------
JSON_FILE = "json_test.txt"   # path to ONE json file
OUT_IMG   = "debug_landmarks.png"
IMG_SIZE  = 512
# ------------------------

with open(JSON_FILE, "r") as f:
    data = json.load(f)

# Pick the first hand
landmarks = data["landmarks"][0]  # shape: (21, 2)
label = data["labels"][0]

# Create blank image
img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# Draw landmarks
for (x, y) in landmarks:
    px = int(x * IMG_SIZE)
    py = int(y * IMG_SIZE)
    cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

# Draw connections (MediaPipe-style)
connections = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

for a, b in connections:
    x1, y1 = landmarks[a]
    x2, y2 = landmarks[b]
    p1 = (int(x1 * IMG_SIZE), int(y1 * IMG_SIZE))
    p2 = (int(x2 * IMG_SIZE), int(y2 * IMG_SIZE))
    cv2.line(img, p1, p2, (255, 0, 0), 2)

cv2.putText(img, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imwrite(OUT_IMG, img)
print(f"Saved {OUT_IMG}")

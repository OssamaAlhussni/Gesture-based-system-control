import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open camera")
    exit(1)
ret, frame = cap.read()
if not ret:
    print("ERROR: cannot read frame")
else:
    print("OK: webcam opened, frame shape:", frame.shape)
cap.release()

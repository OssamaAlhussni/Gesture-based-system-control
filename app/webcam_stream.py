import cv2

def gen_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield in MJPEG format
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
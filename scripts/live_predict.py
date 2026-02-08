#!/usr/bin/env python3
"""
Live webcam prediction (right-hand only):
- MediaPipe hand detection -> normalize -> KNN predict -> smoothing -> overlay label
Display is mirrored (selfie view) but processing/prediction use the original frame.
Ensure u have the .pkl file in the directory for the program to work correctly
prototype
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actions.registry import ActionManager

action_manager = ActionManager()

from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "knn_gesture.pkl")

SMOOTHING_WINDOW = 7 # previously worked with 5
CONF_THRESH = 0.45
CAM_INDEX = 0



STABLE_FRAMES = 3
# ----------------------------

# Load model + encoder (we saved a tuple (knn, le))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

knn, le = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def mp_landmarks_to_vector(hand_landmarks):
    """Convert MediaPipe hand landmarks to the normalized 42-D vector used for training.
       Assumes right-hand geometry (no mirroring)."""
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)  # (21,2)
    origin = pts[0].copy()
    rel = pts - origin
    dists = np.linalg.norm(rel, axis=1)
    maxd = dists.max()
    if maxd < 1e-6:
        maxd = 1.0
    norm = (rel / maxd).flatten()   # shape (42,)
    return norm


def predict_vector(vec):
    """Return (label, confidence). Assumes the loaded KNN supports predict_proba."""
    probs = knn.predict_proba([vec])[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return label, conf


def main():
    last_gesture = None

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    buf = deque(maxlen=SMOOTHING_WINDOW)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("Starting live prediction. Press ESC to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            # Keep a copy for processing (unflipped) so predictions match training
            proc_frame = frame.copy()
            frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            overlay_text = "No hand"
            overlay_conf = 0.0
            

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Convert landmarks to vector (right-hand assumption)
                vec = mp_landmarks_to_vector(hand_landmarks)

                label, conf = predict_vector(vec)

                # apply confidence threshold
                label_display = label if conf >= CONF_THRESH else "uncertain"

                # append to smoothing buffer and majority-vote
                buf.append(label_display)
                mp_draw.draw_landmarks(proc_frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                vote = Counter(buf).most_common(1)[0][0]
                overlay_text = vote
                overlay_conf = conf
                if conf >= 0.75 and vote != last_gesture: #increased the conf threshold to stabilise predictions
                    action_manager.handle(vote)
                    last_gesture = vote
            else:
                    last_gesture = None



            # mirror the annotated frame for display (selfie view)
            display_frame = cv2.flip(proc_frame, 1)

            # Overlay label + conf (draw on display_frame)
            txt = f"{overlay_text} ({overlay_conf:.2f})"
            cv2.putText(display_frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Live Gesture (press ESC)", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

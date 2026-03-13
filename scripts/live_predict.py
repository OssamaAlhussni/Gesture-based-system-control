
#!/usr/bin/env python3
"""
Live webcam gesture recognition (right-hand only)

Pipeline:
MediaPipe hand detection
→ Convert landmarks to normalized vector
→ KNN prediction
→ Temporal smoothing buffer
→ Stable gesture detection
→ Hold gesture for 3 seconds
→ Trigger mapped system action
→ Lock gesture until user changes it

Display is mirrored (selfie view) but prediction uses the original frame.
"""

import os
import sys
import time

# allow importing modules from project root
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

SMOOTHING_WINDOW = 8

CONF_THRESH = 0.50
ACTION_CONF_THRESH = 0.80

MIN_VOTE_COUNT = 4

HOLD_TIME = 3
ACTION_COOLDOWN = 2

CAM_INDEX = 0

# ----------------------------


# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

knn, le = joblib.load(MODEL_PATH)


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def mp_landmarks_to_vector(hand_landmarks):
    """
    Convert MediaPipe landmarks to normalized 42-D vector.
    """

    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)

    origin = pts[0].copy()
    rel = pts - origin

    dists = np.linalg.norm(rel, axis=1)
    maxd = dists.max()

    if maxd < 1e-6:
        maxd = 1.0

    norm = (rel / maxd).flatten()

    return norm


def predict_vector(vec):
    """
    Run KNN prediction and return label + confidence.
    """

    probs = knn.predict_proba([vec])[0]

    idx = int(np.argmax(probs))

    label = le.inverse_transform([idx])[0]
    conf = float(probs[idx])

    return label, conf


def draw_progress_bar(frame, progress):
    """
    Draw hold progress bar.
    """

    bar_x = 10
    bar_y = 60
    bar_w = 200
    bar_h = 12

    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (255,255,255), 1)

    fill = int(bar_w * progress)

    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + fill, bar_y + bar_h),
                  (0,255,0), -1)


def main():

    # gesture state
    current_gesture = None
    gesture_start_time = None
    gesture_locked = None
    last_action_time = 0

    action_message = None
    action_message_time = 0
    ACTION_MSG_DURATION = 2.0

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # smoothing buffer
    buf = deque(maxlen=SMOOTHING_WINDOW)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("Starting live prediction. Press ESC to quit.")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            proc_frame = frame.copy()

            frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            overlay_text = "No hand"
            overlay_conf = 0.0
            progress = 0

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                vec = mp_landmarks_to_vector(hand_landmarks)

                label, conf = predict_vector(vec)

                # show uncertain if confidence too low
                if conf >= CONF_THRESH:
                    label_display = label
                else:
                    label_display = "uncertain"

                # ignore uncertain predictions in smoothing
                if label_display != "uncertain":
                    buf.append(label_display)

                if len(buf) > 0:

                    vote = Counter(buf).most_common(1)[0][0]
                    vote_count = buf.count(vote)

                    overlay_text = vote
                    overlay_conf = conf

                    mp_draw.draw_landmarks(
                        proc_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    current_time = time.time()

                    # gesture must be stable
                    if conf >= ACTION_CONF_THRESH and vote_count >= MIN_VOTE_COUNT:

                        # new gesture detected
                        if vote != current_gesture:

                            current_gesture = vote
                            gesture_start_time = current_time

                        else:

                            if gesture_start_time:

                                held_time = current_time - gesture_start_time

                                progress = min(held_time / HOLD_TIME, 1.0)

                                # trigger action after hold
                                if held_time >= HOLD_TIME:

                                    if (vote != gesture_locked and
                                        current_time - last_action_time > ACTION_COOLDOWN):

                                        action_manager.handle(vote)

                                        # set feedback message
                                        if vote == "index_point":
                                            action_message = "Opening Google Maps"
                                        elif vote == "fist":
                                            action_message = "Mute/Unmute Toggled"
                                        else:
                                            action_message = f"Action: {vote}"

                                        action_message_time = current_time

                                        last_action_time = current_time
                                        gesture_locked = vote

                                    gesture_start_time = None

                    else:
                        current_gesture = None
                        gesture_start_time = None

            else:
                # reset when hand disappears
                current_gesture = None
                gesture_start_time = None
                gesture_locked = None

            # mirror display for selfie view
            display_frame = cv2.flip(proc_frame, 1)

            txt = f"{overlay_text} ({overlay_conf:.2f})"

            cv2.putText(display_frame,
                        txt,
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2)

            # small gesture instructions
            cv2.putText(display_frame,
                        "Index -> Maps",
                        (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200,200,200),
                        1)

            cv2.putText(display_frame,
                        "Fist -> Mute",
                        (10,105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200,200,200),
                        1)

            # show lock message if gesture already triggered
            if overlay_text == gesture_locked:

                cv2.putText(display_frame,
                            "Gesture locked - change gesture",
                            (10,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0,200,255),
                            1)
            if action_message:

                if time.time() - action_message_time < ACTION_MSG_DURATION:

                    text_size = cv2.getTextSize(action_message,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1.0,
                                                2)[0]

                    text_x = (display_frame.shape[1] - text_size[0]) // 2
                    text_y = display_frame.shape[0] // 2

                    cv2.putText(display_frame,
                                action_message,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0,255,0),
                                2)

                else:
                    action_message = None

            # draw hold progress bar
            if progress > 0:
                draw_progress_bar(display_frame, progress)

            cv2.imshow("Live Gesture (press ESC)", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

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

from actions.registry import ActionManager  # <-- import only ActionManager
action_manager = ActionManager()           # <-- manager holds ppt_mode now

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


def draw_centered_text(frame, text, font_scale=1.0, color=(0,255,0), thickness=2, alpha=0.5, padding=15, line_spacing=10):
    """
    Draw multi-line text centered on the frame with a semi-transparent background.
    Each line appears under the previous one.
    """
    lines = text.split("\n")
    
    # Calculate total height of all lines
    line_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]
    total_h = sum([h for w, h in line_sizes]) + line_spacing * (len(lines)-1)
    
    # Starting y-coordinate for first line to vertically center all lines
    start_y = (frame.shape[0] - total_h) // 2
    
    # Draw rectangle behind all lines
    max_w = max([w for w, h in line_sizes])
    text_x = (frame.shape[1] - max_w) // 2
    rect_start = (text_x - padding, start_y - padding)
    rect_end = (text_x + max_w + padding, start_y + total_h + padding)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, rect_start, rect_end, (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw each line
    y = start_y
    for line, (w, h) in zip(lines, line_sizes):
        cv2.putText(frame, line, (text_x, y + h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += h + line_spacing

def main():

    # gesture state
    current_gesture = None
    gesture_start_time = None
    gesture_locked = None
    last_action_time = 0

    action_message = None
    action_message_time = 0
    ACTION_MSG_DURATION = 2.0

    post_ppt_cooldown_time = 0
    POST_PPT_COOLDOWN = 3.0  # ignore gestures for 3 seconds after exiting PPT

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

            current_time = time.time()  # get current time once per loop

            ignore_action = current_time < post_ppt_cooldown_time  # skip gestures if within post-PPT cooldown

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

                                # ---------- ACTION TRIGGER ----------

                                if held_time >= HOLD_TIME:

                                    allow_repeat = action_manager.ppt_mode  # allow repeats in PPT mode

                                    if (vote != gesture_locked or allow_repeat) and (current_time - last_action_time > ACTION_COOLDOWN):

                                        if ignore_action:
                                            # reset hold but skip action
                                            gesture_start_time = None
                                        else:
                                            result = action_manager.handle(vote)
                                            if result == "reset_lock":
                                                gesture_locked = None
                                                # start cooldown after exiting PPT
                                                post_ppt_cooldown_time = current_time + POST_PPT_COOLDOWN

                                        # set feedback message depending on mode
                                        if action_manager.ppt_mode:
                                            if vote == "index_point":
                                                action_message = "Next Slide"
                                            elif vote == "thumbs_up":
                                                action_message = "Previous Slide"
                                            elif vote == "open_palm":
                                                action_message = "Exiting PPT Mode"
                                            elif vote == "peace":
                                                action_message = "Launching PowerPoint"
                                            else:
                                                action_message = f"Action: {vote}"
                                        else:
                                            # normal actions
                                            if vote == "index_point":
                                                action_message = "Opening Google Maps"
                                            elif vote == "fist":
                                                action_message = "Mute/Unmute Toggled"
                                            elif vote == "open_palm":
                                                action_message = ''
                                            elif vote == "peace":
                                                action_message = "Launching PowerPoint"
                                            else:
                                                action_message = f"Action: {vote}"

                                        action_message_time = current_time
                                        last_action_time = current_time
                                        gesture_locked = vote if not action_manager.ppt_mode else None  # allow repeats in PPT mode

                                        # exit gesture hold
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

            # overlay current gesture + confidence
            txt = f"{overlay_text} ({overlay_conf:.2f})"
            cv2.putText(display_frame,
                        txt,
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2)

            # show lock message if gesture already triggered
            if overlay_text == gesture_locked:
                cv2.putText(display_frame,
                            "Gesture locked - change gesture",
                            (10,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0,200,255),
                            1)

            # show action message centered
            if action_message:
                if time.time() - action_message_time < ACTION_MSG_DURATION:
                    draw_centered_text(display_frame, action_message, font_scale=0.9)
                else:
                    action_message = None

            # show open palm mappings if gesture_locked is open_palm and NOT in PPT and NOT in cooldown
            if not action_manager.ppt_mode and gesture_locked == "open_palm" and not ignore_action:
                draw_centered_text(display_frame,
                                "Index -> Maps\nFist -> Mute\nOpen Palm -> Show Mapping",
                                font_scale=0.8,
                                color=(0,255,0))
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
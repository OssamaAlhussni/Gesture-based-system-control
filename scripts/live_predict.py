#!/usr/bin/env python3
"""
Main entry script
Live webcam gesture recognition (right-hand only)
Runs Flask dashboard in background thread on http://localhost:5050
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actions.registry import ActionManager
from actions.voice_assistant import VoiceAssistant
from actions.sos_whatsapp import SosWhatsApp
import app.shared_state as shared_state
from app.dashboard import run as run_dashboard      #Flask

action_manager = ActionManager()
voice_assistant = VoiceAssistant(action_manager)
sos             = SosWhatsApp()

from collections import Counter, deque
import cv2
import joblib
import mediapipe as mp
import numpy as np


#CONFIG
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(BASE_DIR)
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "knn_gesture.pkl")

SMOOTHING_WINDOW   = 8
CONF_THRESH        = 0.50
ACTION_CONF_THRESH = 0.80
MIN_VOTE_COUNT     = 4
ACTION_COOLDOWN    = 2
CAM_INDEX          = 0
#


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

knn, le = joblib.load(MODEL_PATH)
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def mp_landmarks_to_vector(hand_landmarks):
    pts    = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    origin = pts[0].copy()
    rel    = pts - origin
    dists  = np.linalg.norm(rel, axis=1)
    maxd   = dists.max()
    if maxd < 1e-6: maxd = 1.0
    return (rel / maxd).flatten()


def predict_vector(vec):
    probs = knn.predict_proba([vec])[0]
    idx   = int(np.argmax(probs))
    return le.inverse_transform([idx])[0], float(probs[idx])


def draw_progress_bar(frame, progress):
    bx, by, bw, bh = 10, 60, 200, 12
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * progress), by + bh), (0, 255, 0), -1)


def draw_centered_text(frame, text, font_scale=1.0, color=(0, 255, 0),
                        thickness=2, alpha=0.5, padding=15, line_spacing=10):
    lines      = text.split("\n")
    line_sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for l in lines]
    total_h    = sum(h for w, h in line_sizes) + line_spacing * (len(lines) - 1)
    start_y    = (frame.shape[0] - total_h) // 2
    max_w      = max(w for w, h in line_sizes)
    text_x     = (frame.shape[1] - max_w) // 2
    overlay    = frame.copy()
    cv2.rectangle(overlay, (text_x - padding, start_y - padding),
                  (text_x + max_w + padding, start_y + total_h + padding), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    y = start_y
    for line, (w, h) in zip(lines, line_sizes):
        cv2.putText(frame, line, (text_x, y + h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += h + line_spacing


def draw_legend(frame, ppt_mode):
    if ppt_mode:
        lines = [
            "  -- PPT MODE --",
            "Index Point  ->  Next Slide",
            "Thumbs Up    ->  Previous Slide",
            "Open Palm    ->  Exit PPT",
        ]
    else:
        lines = [
            "  -- GESTURE MAP --",
            "Index Point  ->  Google Maps",
            "Fist         ->  Mute / Unmute",
            "Open Palm    ->  SOS WhatsApp",
            "Peace        ->  PowerPoint",
            "Thumbs Up    ->  Voice Assistant",
        ]
    font, fs, th, lh, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1, 22, 8
    total_h = len(lines) * lh + pad * 2
    max_w   = max(cv2.getTextSize(l, font, fs, th)[0][0] for l in lines)
    x0 = 10
    y0 = frame.shape[0] - total_h - 10
    ov = frame.copy()
    cv2.rectangle(ov, (x0 - pad, y0 - pad), (x0 + max_w + pad, y0 + total_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (200, 200, 200)
        cv2.putText(frame, line, (x0, y0 + i * lh + lh), font, fs, color, th)


def main():

    # Start Flask dashboard in background
    flask_thread = threading.Thread(target=run_dashboard, daemon=True)
    flask_thread.start()
    print("Dashboard running at http://localhost:5050")

    current_gesture    = None
    gesture_start_time = None
    gesture_locked     = None
    last_action_time   = 0
    action_message     = None
    action_message_time= 0
    ACTION_MSG_DURATION= 2.0
    post_ppt_cooldown_time = 0
    POST_PPT_COOLDOWN  = 3.0

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

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

            proc_frame   = frame.copy()
            frame_rgb    = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results      = hands.process(frame_rgb)

            overlay_text = "No hand"
            overlay_conf = 0.0
            progress     = 0
            current_time = time.time()
            ignore_action= current_time < post_ppt_cooldown_time

            # read hold_time from shared state so settings panel can update it live
            HOLD_TIME = shared_state.get_state().get("hold_time", 3)

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]
                vec            = mp_landmarks_to_vector(hand_landmarks)
                label, conf    = predict_vector(vec)
                label_display  = label if conf >= CONF_THRESH else "uncertain"

                if label_display != "uncertain":
                    buf.append(label_display)

                if len(buf) > 0:
                    vote       = Counter(buf).most_common(1)[0][0]
                    vote_count = buf.count(vote)
                    overlay_text = vote
                    overlay_conf = conf

                    mp_draw.draw_landmarks(proc_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if conf >= ACTION_CONF_THRESH and vote_count >= MIN_VOTE_COUNT:

                        if vote != current_gesture:
                            current_gesture    = vote
                            gesture_start_time = current_time
                        else:
                            if gesture_start_time:
                                held_time = current_time - gesture_start_time
                                progress  = min(held_time / HOLD_TIME, 1.0)

                                if held_time >= HOLD_TIME:
                                    allow_repeat = action_manager.ppt_mode

                                    if (vote != gesture_locked or allow_repeat) and \
                                       (current_time - last_action_time > ACTION_COOLDOWN):

                                        was_ppt_mode = action_manager.ppt_mode

                                        if ignore_action:
                                            gesture_start_time = None
                                        else:
                                            if not action_manager.ppt_mode:
                                                if vote == "thumbs_up":
                                                    voice_assistant.trigger()
                                                elif vote == "open_palm":
                                                    sos.trigger()
                                                else:
                                                    result = action_manager.handle(vote)
                                                    if result == "reset_lock":
                                                        gesture_locked = None
                                                        post_ppt_cooldown_time = current_time + POST_PPT_COOLDOWN
                                            else:
                                                result = action_manager.handle(vote)
                                                if result == "reset_lock":
                                                    gesture_locked = None
                                                    post_ppt_cooldown_time = current_time + POST_PPT_COOLDOWN

                                        # action message and shared state update
                                        if was_ppt_mode:
                                            msgs = {
                                                "index_point": "Next Slide",
                                                "thumbs_up"  : "Previous Slide",
                                                "open_palm"  : "Exiting PPT Mode",
                                                "peace"      : "Launching PowerPoint",
                                            }
                                        else:
                                            msgs = {
                                                "index_point": "Opening Google Maps",
                                                "fist"       : "Mute/Unmute Toggled",
                                                "open_palm"  : "SOS Sent!",
                                                "peace"      : "Launching PowerPoint",
                                                "thumbs_up"  : "Listening...",
                                            }
                                        action_message = msgs.get(vote, f"Action: {vote}")
                                        shared_state.update_state(last_action=action_message)

                                        action_message_time = current_time
                                        last_action_time    = current_time
                                        gesture_locked      = vote if not action_manager.ppt_mode else None
                                        gesture_start_time  = None

                    else:
                        current_gesture    = None
                        gesture_start_time = None

            else:
                current_gesture    = None
                gesture_start_time = None
                gesture_locked     = None

            # update shared state for dashboard
            shared_state.update_state(
                current_gesture=overlay_text,
                confidence=overlay_conf,
            )

            display_frame = cv2.flip(proc_frame, 1)

            cv2.putText(display_frame,
                        f"{overlay_text} ({overlay_conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if overlay_text == gesture_locked:
                cv2.putText(display_frame,
                            "Gesture locked - change gesture",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            if action_message:
                if time.time() - action_message_time < ACTION_MSG_DURATION:
                    draw_centered_text(display_frame, action_message, font_scale=0.9)
                else:
                    action_message = None

            if action_manager.ppt_mode:
                lines = ["Index -> Next Slide", "Thumbs Up -> Previous Slide", "Open Palm -> Exit"]
                y = 10
                for line in lines:
                    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.putText(display_frame, line,
                                (display_frame.shape[1] - tw - 10, y + th),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    y += th + 5

            #draw_legend(display_frame, action_manager.ppt_mode)

            if progress > 0:
                draw_progress_bar(display_frame, progress)

            # share frame with Flask
            with shared_state.frame_lock:
                shared_state.latest_frame = display_frame.copy()

            cv2.imshow("Live Gesture (press ESC)", display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# import cv2, mediapipe as mp
# mp_hands = mp.solutions.hands
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("ERROR: cannot open camera"); exit(1)
# ret, frame = cap.read()
# cap.release()
# if not ret:
#     print("ERROR: cannot read frame"); exit(1)
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
#     res = hands.process(img)
#     if not res.multi_hand_landmarks:
#         print("No hand detected (try moving hand into frame)")
#     else:
#         print("Detected", len(res.multi_hand_landmarks), "hand(s)")
#         # show first landmark (x,y,z of first point)
#         lm = res.multi_hand_landmarks[0].landmark[0]
#         print("first landmark:", lm.x, lm.y, lm.z)


import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # for drawing landmarks

cap = cv2.VideoCapture(0)  # correct webcam index

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't read frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            print(f"Detected {len(results.multi_hand_landmarks)} hand(s)")
            # Draw landmarks for each hand
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            print("No hands detected")

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
            break

cap.release()
cv2.destroyAllWindows()


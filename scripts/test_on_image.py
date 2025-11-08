import cv2, mediapipe as mp, numpy as np, sys, os
mp_hands = mp.solutions.hands

img_name = "1000.jpg"

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "..", img_name) 
print(img_path)
img = cv2.imread(img_path)
if img is None:
    print("ERROR: place a test.jpg in project root")
    sys.exit(1)

    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        print(f'No hand detected in {img_name}')
    else:
        print("Detected", len(res.multi_hand_landmarks), "hand(s)")
        coords = [[(p.x, p.y, p.z) for p in lm.landmark] for lm in res.multi_hand_landmarks]
        print("First landmark of first hand:", coords[0][0])

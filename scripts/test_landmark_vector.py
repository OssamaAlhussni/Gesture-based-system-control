import cv2, mediapipe as mp, numpy as np, os
mp_hands = mp.solutions.hands

def extract_norm_vector(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark])  # (21,3)
        coords = coords - coords[0]   # translate wrist to origin
        maxv = np.max(np.abs(coords))
        if maxv > 0:
            coords = coords / maxv
        return coords.flatten()

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "..", "test_image.jpg") 
vec = extract_norm_vector(img_path)
if vec is None:
    print("No hand in test.jpg")
else:
    print("Vector length:", len(vec), "first 6 values:", vec[:6])

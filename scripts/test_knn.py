
# testing for 1 random sample at a time

# import joblib
# import numpy as np
# import csv
# import random
# from sklearn.metrics import classification_report
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# root = os.path.dirname(BASE_DIR)
# MODEL_PATH = os.path.join(root,"models","knn_gesture.pkl")
# DATA_PATH = os.path.join(BASE_DIR,"data","gestures_all.csv")

# knn, le = joblib.load(MODEL_PATH)

# samples = []
# y_true = []
# y_pred = []

# with open(DATA_PATH, newline="") as f:
#     reader = csv.reader(f)
#     header = next(reader)
#     for row in reader:
#         label = row[0]
#         vec = np.array([float(v) for v in row[1:]])
#         samples.append((label, vec))

# # Random test
# label, vec = random.choice(samples)
# pred = knn.predict([vec])[0]

# print("True label:", label)
# print("Predicted :", le.inverse_transform([pred])[0])



# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred))


#====================================================================================================================
# testing for all samples at the same time

import joblib
import numpy as np
import csv
from sklearn.metrics import classification_report
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(root, "models", "knn_gesture.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "gestures_all.csv")

knn, le = joblib.load(MODEL_PATH)

X = []
y_true = []

with open(DATA_PATH, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        label = row[0]
        vec = np.array([float(v) for v in row[1:]])
        X.append(vec)
        y_true.append(label)

# Predict all samples
y_pred_encoded = knn.predict(X)
y_pred = le.inverse_transform(y_pred_encoded)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

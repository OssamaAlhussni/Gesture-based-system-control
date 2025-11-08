import csv
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "gestures_all.csv")
print("Base: ",BASE_DIR)
#DATA_PATH = "data/gestures_all.csv"
#MODEL_OUT = "../models/knn_gesture.pkl"
BASE_MODEL = os.path.dirname(BASE_DIR)
MODEL_OUT = os.path.join(BASE_MODEL,"models","knn_gesture.pkl")
print(MODEL_OUT)
print("Using CSV at:", DATA_PATH)
print("Exists:", os.path.exists(DATA_PATH))

X = []
y = []
# Load CSV
with open(DATA_PATH, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        y.append(row[0])                 # label
        X.append([float(v) for v in row[1:]])  # landmarks

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Choose K
k = 5   # small dataset → small k
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")

knn.fit(X, y_enc)

# Save model + encoder
joblib.dump((knn, le), MODEL_OUT)

print("✅ KNN trained")
print("Samples:", len(X))
print("Classes:", list(le.classes_))
print("Model saved to:", MODEL_OUT)
print("Model saved to:", MODEL_OUT)

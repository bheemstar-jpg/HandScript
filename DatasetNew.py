import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = r'C:\Users\roanm\Desktop\Cmpe 246 project\NewPhotos'
data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = hand.landmark

            x_vals = [lm.x for lm in landmarks]
            y_vals = [lm.y for lm in landmarks]
            z_vals = [lm.z for lm in landmarks]

            min_x, min_y = min(x_vals), min(y_vals)
            max_x, max_y = max(x_vals), max(y_vals)

            width = max_x - min_x
            height = max_y - min_y

            if width == 0 or height == 0:
                continue  # Avoid division by zero

            sample = []
            for lm in landmarks:
                norm_x = (lm.x - min_x) / width
                norm_y = (lm.y - min_y) / height
                sample.extend([norm_x, norm_y, lm.z])  # Include z

            if len(sample) == 63:  # 21 landmarks × 3
                data.append(sample)
                labels.append(int(label))

# Save data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("✅ Data saved with", len(data), "samples.")

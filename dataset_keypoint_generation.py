import cv2
import mediapipe as mp
import numpy as np
import csv
import os 
print("Notebook reading from:",os.path.abspath("keypoint.csv"))

# ==============================
# 🔹 Enter the label here
# ==============================
label = input("Enter gesture label (A, B, C, etc): ")

# ==============================
# 🔹 Setup Mediapipe
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ==============================
# 🔹 Start Camera
# ==============================
cap = cv2.VideoCapture(0)

print("Press 's' to save sample")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==============================
            # 🔹 Extract keypoints
            # ==============================
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)

            keypoints = np.array(keypoints)

            # ==============================
            # 🔹 Wrist-relative normalization
            # ==============================
            wrist_x = keypoints[0]
            wrist_y = keypoints[1]

            for i in range(0, len(keypoints), 2):
                keypoints[i]   -= wrist_x
                keypoints[i+1] -= wrist_y

            # ==============================
            # 🔹 Scale normalization
            # ==============================
            max_val = np.max(np.abs(keypoints))
            if max_val != 0:
                keypoints = keypoints / max_val

            keypoints = keypoints.tolist()

            # ==============================
            # 🔹 Save on pressing 's'
            # ==============================
            key = cv2.waitKey(1)

            if key == ord('s'):
                with open(r"C:\Users\harsh\OneDrive\Desktop\ISL_PROJECT\02_repo\keypoint.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + keypoints)

                print(f"Saved sample for {label}")

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Dataset Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
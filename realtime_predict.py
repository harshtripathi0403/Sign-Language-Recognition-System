import cv2
import numpy as np
import joblib
import mediapipe as mp

# ==========================================
# Load trained model and encoder
# ==========================================
model = joblib.load("02_repo/isl_model.pkl")
encoder = joblib.load("02_repo/label_encoder.pkl")

# ==========================================
# Mediapipe setup
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# Start camera
# ==========================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==========================================
            # Extract raw keypoints (same as dataset)
            # ==========================================
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)

            keypoints = np.array(keypoints)

            # ==========================================
            # Wrist-relative normalization
            # ==========================================
            wrist_x = keypoints[0]
            wrist_y = keypoints[1]

            for i in range(0, len(keypoints), 2):
                keypoints[i] -= wrist_x
                keypoints[i+1] -= wrist_y

            # ==========================================
            # Scale normalization
            # ==========================================
            max_val = np.max(np.abs(keypoints))
            if max_val != 0:
                keypoints = keypoints / max_val

            # Reshape for model
            keypoints = keypoints.reshape(1, -1)

            # ==========================================
            # Prediction
            # ==========================================
            probs = model.predict_proba(keypoints)
            prediction = model.predict(keypoints)
            label = encoder.inverse_transform(prediction)[0]

           

            # Display result
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("ISL Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Proper shutdown
cap.release()
cv2.destroyAllWindows()
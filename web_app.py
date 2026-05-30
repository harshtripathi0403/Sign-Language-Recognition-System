import cv2
import numpy as np
import joblib
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

# Load Models
try:
    model = joblib.load("02_repo/isl_model.pkl")
    encoder = joblib.load("02_repo/label_encoder.pkl")
except Exception as e:
    print("Error loading models, make sure you are running from root:", e)
    model = None
    encoder = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# DO NOT init camera automatically!
cap = None
is_camera_on = False

# Global State
current_state = {
    "prediction": "-",
    "confidence": 0.0,
    "camera_on": False
}

def generate_frames():
    global current_state, is_camera_on, cap
    
    while True:
        if not is_camera_on or cap is None or not cap.isOpened():
            # Camera off - yield a blank/black frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Offline", (190, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            cv2.putText(frame, "Click 'Start Camera' below", (140, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        # Mediapipe Prediction Logic
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        prediction_made = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append(lm.x)
                    keypoints.append(lm.y)
                    
                keypoints = np.array(keypoints)
                wrist_x = keypoints[0]
                wrist_y = keypoints[1]
                
                for i in range(0, len(keypoints), 2):
                    keypoints[i] -= wrist_x
                    keypoints[i+1] -= wrist_y
                    
                max_val = np.max(np.abs(keypoints))
                if max_val != 0:
                    keypoints = keypoints / max_val
                    
                keypoints = keypoints.reshape(1, -1)
                
                if model and encoder:
                    probs = model.predict_proba(keypoints)
                    prediction = model.predict(keypoints)
                    label = encoder.inverse_transform(prediction)[0]
                    max_prob = np.max(probs) * 100
                    
                    current_state["prediction"] = str(label)
                    current_state["confidence"] = float(max_prob)
                    prediction_made = True

        if not prediction_made:
            current_state["prediction"] = "-"
            current_state["confidence"] = 0.0
            
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def state():
    return jsonify(current_state)

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera_api():
    global is_camera_on, cap, current_state
    if is_camera_on:
        # Turn off
        if cap and cap.isOpened():
            cap.release()
        is_camera_on = False
        current_state["camera_on"] = False
        current_state["prediction"] = "-"
        current_state["confidence"] = 0.0
    else:
        # Turn on
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        is_camera_on = True
        current_state["camera_on"] = True
    
    return jsonify({"status": "ok", "camera_on": is_camera_on})

if __name__ == "__main__":
    print("Server started on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

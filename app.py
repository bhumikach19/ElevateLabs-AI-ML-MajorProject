# app.py

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
# from playsound import playsound # Uncomment if you want sound alerts in web app, might be tricky

app = Flask(__name__)

# --- Configuration (same as before) ---
MODEL_PATH = r'C:\Users\BHUMIKA\Desktop\Elevate Labs\Major Project\mask_detection_model.h5'
FACE_CASCADE_PATH = r'C:\Users\BHUMIKA\Desktop\Elevate Labs\Major Project\haarcascade_frontalface_default.xml'
IMG_HEIGHT = 150
IMG_WIDTH = 150
ALERT_SOUND_PATH = 'alert.wav' # Optional, might not work well in web app

# --- Load Model and Haar Cascade (load once when app starts) ---
try:
    model = load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set to None if loading fails
    face_cascade = None # Set to None if loading fails

try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar Cascade XML file: {FACE_CASCADE_PATH}")
    print(f"Successfully loaded Haar Cascade from {FACE_CASCADE_PATH}")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    face_cascade = None # Set to None if loading fails


# --- Video Capture (initialize globally or within generator) ---
# It's often better to initialize inside the generator for Flask to manage connections better
# cap = cv2.VideoCapture(0)


def generate_frames():
    cap = cv2.VideoCapture(0) # Initialize webcam here

    if not cap.isOpened():
        print("Error: Could not open video stream for Flask app.")
        return

    # Alert control variables for this stream
    # alert_active = False # Not strictly needed if we just draw text
    # last_alert_time = time.time()
    # alert_cooldown = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if model is None or face_cascade is None:
            # Display an error message if model or cascade failed to load
            cv2.putText(frame, "Error: Model or Cascade not loaded!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue # Skip detection if model/cascade not available

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        no_mask_detected_this_frame = False

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            prediction = model.predict(face_input)[0][0]

            if prediction < 0.5:
                label = "Mask Worn"
                color = (0, 255, 0)
            else:
                label = "NO MASK!"
                color = (0, 0, 255)
                no_mask_detected_this_frame = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Alert Logic for Flask (visual only) ---
        if no_mask_detected_this_frame:
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, "ALERT: MASK REQUIRED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            # Sound alerts (`playsound`) are generally not recommended directly in Flask
            # because they'd play on the server, not the client's browser.
            # For client-side sound, you'd need JavaScript.

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template(r'C:\Users\BHUMIKA\Desktop\Elevate Labs\Major Project\template\index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
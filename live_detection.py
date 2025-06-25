# python_code_step_4.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model # For loading the trained model
import os

# --- Configuration ---
# Path to your trained Keras model (from Step 3)
MODEL_PATH = r'C:\Users\BHUMIKA\Desktop\Elevate Labs\Major Project\mask_detection_model.h5'

# Path to the Haar Cascade XML file
# Make sure this file is in the same directory as your script or provide the full path
FACE_CASCADE_PATH = r'C:\Users\BHUMIKA\Desktop\Elevate Labs\Major Project\haarcascade_frontalface_default.xml'

# Image dimensions your model was trained on
IMG_HEIGHT = 150
IMG_WIDTH = 150

# --- Load the trained model ---
try:
    model = load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'mask_detection_model.keras' exists and is a valid Keras model file.")
    exit()

# --- Load Haar Cascade for face detection ---
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar Cascade XML file: {FACE_CASCADE_PATH}")
    print(f"Successfully loaded Haar Cascade from {FACE_CASCADE_PATH}")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    print("Please ensure 'haarcascade_frontalface_default.xml' is in the correct path.")
    exit()

# --- Initialize webcam ---
cap = cv2.VideoCapture(0) # 0 for default webcam, change if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

# --- Real-time detection loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert frame to grayscale for Haar Cascade (more efficient)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # (scaleFactor, minNeighbors, minSize) can be tuned for better detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract the face region from the original color frame
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face ROI for the CNN model
        # Resize to model's input size
        face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
        # Convert to RGB (OpenCV reads BGR by default) - essential if your model was trained on RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        face_normalized = face_rgb / 255.0
        # Add batch dimension (1, IMG_HEIGHT, IMG_WIDTH, 3)
        face_input = np.expand_dims(face_normalized, axis=0)

        # Make prediction
        prediction = model.predict(face_input)[0][0]

        # Determine label and color based on prediction
        if prediction > 0.5: # Threshold for 'with_mask' (adjust as needed)
            label = "Mask Worn"
            color = (0, 255, 0) # Green for mask
        else:
            label = "NO MASK!"
            color = (0, 0, 255) # Red for no mask

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.")
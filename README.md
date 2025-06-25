# Face Mask Detection Web App

This project is a real-time face mask detection system using a Convolutional Neural Network (CNN) and OpenCV, with a web interface powered by Flask. It detects whether people are wearing masks using your webcam.

## Features
- Real-time face mask detection using webcam
- Web interface for live video streaming and detection
- Trained with Keras/TensorFlow
- Uses OpenCV for face detection

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Files
- `app.py`: Flask web app for live mask detection
- `live_detection.py`: Standalone script for real-time detection (no web interface)
- `EL_Major.ipynb`: Jupyter notebook for model training and experimentation
- `mask_detection_model.h5`: Trained Keras model file
- `haarcascade_frontalface_default.xml`: Haar Cascade for face detection
- `requirements.txt`: Python dependencies

## Usage
### 1. Web App
Start the Flask server:
```bash
python app.py
```
Visit `http://localhost:5000` in your browser to see the live detection.

### 2. Standalone Detection
Run the detection script:
```bash
python live_detection.py
```
A window will open showing webcam feed with mask detection.

## Notes
- Ensure your webcam is connected and accessible.
- The model and Haar Cascade paths in the scripts may need to be updated to match your local file locations.
- For training or retraining the model, use the Jupyter notebook (`EL_Major.ipynb`).

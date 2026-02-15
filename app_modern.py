"""
app_modern.py — Flask backend for the ASL Translation System.

Integrates the existing SignModel and LandmarkExtractor with a modern web UI.
"""

import cv2
import json
import time
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify

from LandmarkExtractor import LandmarkExtractor

try:
    from SignModel import load_model, load_labels, predict_sign, PredictionStabilizer
    HAS_MODEL_SUPPORT = True
except ImportError:
    print("[WARNING] TensorFlow/SignModel not available. Running in UI-only mode.")
    HAS_MODEL_SUPPORT = False
    
    def load_model(path): return None
    def load_labels(path): return {}
    def predict_sign(*args, **kwargs): return []
    
    class PredictionStabilizer:
        def __init__(self, *args, **kwargs): pass
        def add_prediction(self, *args, **kwargs): return None
        def reset(self): pass

app = Flask(__name__)

# ──────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────

class ASLSystem:
    def __init__(self):
        self.extractor = LandmarkExtractor()
        self.model = None
        self.labels = None
        self.frame_buffer = deque(maxlen=30)
        # Conditional stabilizer
        self.stabilizer = PredictionStabilizer(min_confidence=0.90, stability_frames=8) if HAS_MODEL_SUPPORT else None
        self.sentence = []
        self.latest_prediction = {"sign": "--", "confidence": 0.0}
        self.is_running = False

    def load_resources(self):
        if not HAS_MODEL_SUPPORT:
            print("[ERROR] Cannot load resources: TensorFlow not installed.")
            return False

        try:
            self.model = load_model("models/asl_model.h5")
            self.labels = load_labels("labels.json")
            print("[OK] Model and labels loaded successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading resources: {e}")
            return False

    def process_frame(self, frame):
        # 1. Extract landmarks (draw them on frame for feedback)
        landmarks, annotated_frame = self.extractor.process_frame_with_drawing(frame)
        
        # 2. Add to buffer
        self.frame_buffer.append(landmarks)

        # 3. Predict if buffer is full
        if HAS_MODEL_SUPPORT and len(self.frame_buffer) == 30 and self.model:
            sequence = np.array(list(self.frame_buffer))
            results = predict_sign(self.model, sequence, self.labels, top_k=1)
            
            if results:
                sign, conf = results[0]
                self.latest_prediction = {"sign": sign, "confidence": float(conf)}

                # Stabilize
                stable_sign = self.stabilizer.add_prediction(sign, conf)
                if stable_sign:
                    if not self.sentence or self.sentence[-1] != stable_sign:
                        self.sentence.append(stable_sign)
                        if len(self.sentence) > 10:
                            self.sentence.pop(0)
            else:
                self.latest_prediction = {"sign": "--", "confidence": 0.0}
        
        return annotated_frame

asl_system = ASLSystem()

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_detection():
    if not asl_system.model:
        success = asl_system.load_resources()
        if not success and HAS_MODEL_SUPPORT:
            return jsonify({"status": "error", "message": "Failed to load model"}), 500
    
    asl_system.is_running = True
    asl_system.sentence = []
    # If no model support, we just start without loading model
    return jsonify({"status": "started"})

@app.route('/stop')
def stop_detection():
    asl_system.is_running = False
    return jsonify({"status": "stopped"})

@app.route('/reset')
def reset():
    asl_system.sentence = []
    asl_system.frame_buffer.clear()
    if asl_system.stabilizer:
        asl_system.stabilizer.reset()
    asl_system.latest_prediction = {"sign": "--", "confidence": 0.0}
    return jsonify({"status": "reset"})

@app.route('/status')
def status():
    return jsonify({
        "prediction": asl_system.latest_prediction,
        "sentence": " ".join(asl_system.sentence),
        "is_running": asl_system.is_running
    })

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    while True:
        if not asl_system.is_running:
            # If stopped, send a placeholder or black frame
            # Or just wait to avoid burning CPU
            time.sleep(0.1)
            ret, frame = cap.read() # Keep reading to clear buffer
            if not ret: break
            
            # Optional: Send a "paused" frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Process logic
        processed_frame = asl_system.process_frame(frame)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

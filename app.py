"""
app.py — Full Streamlit application for LSTM-based dynamic sign recognition.

Uses LandmarkExtractor (MediaPipe Holistic, 1662 features) and
SignModel (LSTM) for word/phrase-level ASL detection.

Run with:  streamlit run app.py
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

from LandmarkExtractor import LandmarkExtractor
from SignModel import load_model, load_labels, predict_sign, PredictionStabilizer


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

CUSTOM_CSS = """
<style>
.main-title {
    font-size: 48px;
    font-weight: bold;
    color: #1e88e5;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.prediction-box {
    border-left: 4px solid #1e88e5;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
    border-radius: 0 8px 8px 0;
}
</style>
"""


class ASLTranslationApp:
    """
    Manages LSTM-based sign prediction with a 30-frame buffer,
    stabilization, and sentence accumulation.
    """

    def __init__(self):
        self.frame_buffer = deque(maxlen=30)
        self.stabilizer = PredictionStabilizer(
            min_confidence=0.90,
            stability_frames=10,
        )
        self.sentence = []
        self.model = None
        self.labels = None
        self.extractor = None

    def load_resources(self, model_path, labels_path):
        """Load the LSTM model, labels, and initialize the extractor."""
        self.model = load_model(model_path)
        self.labels = load_labels(labels_path)
        self.extractor = LandmarkExtractor()
        return True

    def process_frame(self, frame):
        """
        Process a single frame:
            1. Extract + draw landmarks.
            2. Append to 30-frame buffer.
            3. When buffer full → predict with LSTM → check stability.
            4. Draw HUD overlay.

        Returns:
            (annotated_frame, prediction, confidence)
        """
        landmarks, annotated = self.extractor.process_frame_with_drawing(
            frame, thickness=2
        )

        self.frame_buffer.append(landmarks)

        prediction = None
        confidence = 0.0

        if len(self.frame_buffer) == 30 and self.model is not None:
            sequence = np.array(list(self.frame_buffer))
            results = predict_sign(self.model, sequence, self.labels, top_k=1)

            if results:
                prediction, confidence = results[0]

                # Check stabilizer.
                stable = self.stabilizer.add_prediction(prediction, confidence)
                if stable:
                    # Add to sentence (max 20 words).
                    if len(self.sentence) < 20:
                        self.sentence.append(stable)

        self.draw_hud(annotated, prediction, confidence)
        return annotated, prediction, confidence

    def draw_hud(self, frame, prediction, confidence):
        """
        Draw a heads-up display overlay on the frame.

        Shows prediction, confidence %, sentence history, and buffer status.
        """
        h, w, _ = frame.shape

        # Semi-transparent black bar at top.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        if prediction:
            # Prediction text.
            color = (0, 200, 0) if confidence >= 0.9 else (0, 180, 255)
            cv2.putText(
                frame,
                prediction.upper(),
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                3,
            )
            # Confidence.
            cv2.putText(
                frame,
                f"{confidence * 100:.1f}%",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Waiting for sign...",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (150, 150, 150),
                2,
            )

        # Sentence on the right.
        sentence_text = " ".join(self.sentence)
        if len(sentence_text) > 60:
            sentence_text = "..." + sentence_text[-57:]
        cv2.putText(
            frame,
            sentence_text,
            (w // 2, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Buffer status.
        buffer_text = f"Buffer: {len(self.frame_buffer)}/30"
        cv2.putText(
            frame,
            buffer_text,
            (w - 180, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )


# ──────────────────────────────────────────────
# Streamlit main
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ASL Translation (LSTM)",
        page_icon="\U0001f91f",
        layout="wide",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="main-title">\U0001f91f ASL Translation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dynamic Sign Recognition with LSTM</div>', unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input("Model path", value="models/asl_model.h5")
        labels_path = st.text_input("Labels path", value="labels.json")
        camera_index = st.selectbox("Camera", [0, 1, 2], index=0)

        load_btn = st.button("Load Model & Labels")

    # Session state for the app instance.
    if "asl_app" not in st.session_state:
        st.session_state.asl_app = ASLTranslationApp()
        st.session_state.resources_loaded = False

    app = st.session_state.asl_app

    if load_btn:
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                app.load_resources(model_path, labels_path)
                st.session_state.resources_loaded = True
                st.sidebar.success("Model and labels loaded!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.error("Model or labels file not found.")

    # ── Main layout ──
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("\U0001f4f9 Live Camera Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("\U0001f3af Detection")
        prediction_placeholder = st.empty()
        st.subheader("\U0001f4ac Sentence")
        sentence_placeholder = st.empty()

    # Instructions expander.
    with st.expander("Usage Tips"):
        st.markdown(
            "1. Load the LSTM model and labels first.\n"
            "2. Click **Start Camera** to begin detection.\n"
            "3. Perform a sign for ~1 second (30 frames).\n"
            "4. The model predicts after filling the 30-frame buffer.\n"
            "5. Stable predictions are added to the sentence."
        )

    # Camera controls.
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        start_btn = st.button(
            "\u25b6\ufe0f Start Camera",
            type="primary",
            disabled=not st.session_state.resources_loaded,
        )
    with ctrl2:
        stop_btn = st.button("\u23f9\ufe0f Stop Camera")

    if start_btn:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                annotated, prediction, confidence = app.process_frame(frame)

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    frame_rgb, channels="RGB", use_container_width=True
                )

                # Update prediction display.
                if prediction:
                    color = "green" if confidence >= 0.9 else "orange"
                    prediction_placeholder.markdown(
                        f'<div class="prediction-box">'
                        f"<b>{prediction.upper()}</b> — {confidence:.0%}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    prediction_placeholder.markdown("Waiting for sign...")

                # Update sentence.
                sentence_placeholder.markdown(
                    " ".join(app.sentence) if app.sentence else "_No signs detected yet_"
                )

                time.sleep(0.01)

            cap.release()
            if app.extractor:
                app.extractor.close()

    # Footer.
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:gray;">'
        "Built with MediaPipe Holistic, TensorFlow/Keras, and Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

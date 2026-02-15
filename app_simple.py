"""
app_simple.py — Primary Streamlit application for ASL letter detection.

Uses MediaPipe Hands + Random Forest for real-time letter recognition.
Run with:  streamlit run app_simple.py
"""

import os
import json
import time
import pickle

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st


# ──────────────────────────────────────────────
# Normalization (must match collect_letters.py)
# ──────────────────────────────────────────────

def normalize_hand_landmarks(hand_landmarks):
    """
    Convert 21 hand landmarks into a 63-feature normalized vector.

    Same normalization as used during data collection:
        1. Wrist-relative shift
        2. Palm-size scaling
        3. Flatten to 63 floats
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    coords = coords - coords[0]

    palm_size = np.linalg.norm(coords[9])
    if palm_size >= 1e-6:
        coords = coords / palm_size

    return coords.flatten().astype(np.float32)


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

@st.cache_resource
def load_letter_model():
    """
    Load the trained Random Forest model and label map.

    Returns:
        (model, labels) or (None, None) if files don't exist.
    """
    model_path = os.path.join("models", "letter_model.pkl")
    labels_path = "letter_labels.json"

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Convert string keys to int.
    labels = {int(k): v for k, v in labels.items()}

    return model, labels


# ──────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="\U0001f91f",
    layout="wide",
)

st.title("\U0001f91f ASL Translation System")
st.markdown("**Sign Language Detector** — Real-time ASL letter recognition")

# Load model.
model, labels = load_letter_model()

# ── Sidebar ──
with st.sidebar:
    st.header("Settings")
    camera_index = st.selectbox("Camera", [0, 1, 2], index=0)

    st.divider()

    if model is not None and labels is not None:
        st.success(f"Model loaded: {len(labels)} letters")
        letter_list = ", ".join(sorted(labels.values()))
        st.caption(f"Letters: {letter_list}")
    else:
        st.warning(
            "No trained model found.\n\n"
            "Run these steps first:\n"
            "1. `python collect_letters.py`\n"
            "2. `python train_letters.py`\n"
            "3. Restart this app"
        )

    st.divider()
    st.markdown("**Landmark Colors**")
    st.markdown("\U0001f534 Red = Left hand")
    st.markdown("\U0001f535 Blue = Right hand")

# ── Main layout ──
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("\U0001f4f9 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("\U0001f3af Detection")
    prediction_placeholder = st.empty()
    st.subheader("\U0001f4ca Stats")
    stats_placeholder = st.empty()

# ── Camera controls ──
ctrl1, ctrl2 = st.columns(2)
with ctrl1:
    start_btn = st.button("\u25b6\ufe0f Start Camera", type="primary")
with ctrl2:
    stop_btn = st.button("\u23f9\ufe0f Stop Camera")

# ── Camera loop ──
if start_btn:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Cannot open webcam. Try a different Camera index in the sidebar.")
    else:
        frame_count = 0
        fps = 0.0
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Lost camera feed.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            pred_text = "No hand detected"
            pred_confidence = 0.0

            if results.multi_hand_landmarks:
                for idx_hand, hand_lm in enumerate(results.multi_hand_landmarks):
                    # Determine hand label (flipped because image is mirrored).
                    hand_label = "Right"
                    if results.multi_handedness:
                        mp_label = results.multi_handedness[idx_hand].classification[0].label
                        hand_label = "Right" if mp_label == "Left" else "Left"

                    # Choose color: Left = red, Right = blue.
                    if hand_label == "Left":
                        color = (0, 0, 255)   # Red in BGR
                    else:
                        color = (255, 0, 0)   # Blue in BGR

                    # Draw landmarks.
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    )

                    # Predict (if model loaded).
                    if model is not None and labels is not None:
                        features = normalize_hand_landmarks(hand_lm)
                        proba = model.predict_proba(features.reshape(1, -1))[0]
                        best_idx = np.argmax(proba)
                        confidence = proba[best_idx]

                        if confidence > 0.5:
                            pred_letter = labels.get(best_idx, "?")
                            pred_text = f"{pred_letter} ({confidence:.0%})"
                            pred_confidence = confidence

                            # Position text above the hand.
                            tip = hand_lm.landmark[12]  # Middle finger tip
                            wrist = hand_lm.landmark[0]
                            text_x = int(tip.x * w)
                            text_y = int(min(wrist.y, tip.y) * h) - 40
                            text_x = max(10, min(text_x, w - 100))
                            text_y = max(30, text_y)

                            # Background rectangle.
                            box_color = (0, 200, 0) if confidence >= 0.8 else (0, 200, 200)
                            label_text = f"{pred_letter} {confidence:.0%}"
                            text_size = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                            )[0]
                            cv2.rectangle(
                                frame,
                                (text_x - 5, text_y - text_size[1] - 10),
                                (text_x + text_size[0] + 5, text_y + 5),
                                (0, 0, 0),
                                -1,
                            )
                            cv2.rectangle(
                                frame,
                                (text_x - 5, text_y - text_size[1] - 10),
                                (text_x + text_size[0] + 5, text_y + 5),
                                box_color,
                                2,
                            )
                            cv2.putText(
                                frame,
                                label_text,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                3,
                            )

            # Convert BGR → RGB for Streamlit.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update prediction sidebar.
            if pred_confidence >= 0.8:
                prediction_placeholder.markdown(
                    f"### :green[{pred_text}]"
                )
            elif pred_confidence > 0.5:
                prediction_placeholder.markdown(
                    f"### :orange[{pred_text}]"
                )
            else:
                prediction_placeholder.markdown(f"### {pred_text}")

            # Update stats.
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            stats_placeholder.markdown(
                f"**FPS:** {fps:.1f}  \n"
                f"**Frames:** {frame_count}  \n"
                f"**Model:** {'Loaded' if model else 'Not loaded'}"
            )

            time.sleep(0.03)

        cap.release()
        hands.close()

# ── Footer ──
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:gray;">'
    "ASL Sign Language Detector &mdash; Built with MediaPipe Hands + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)

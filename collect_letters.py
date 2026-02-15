"""
collect_letters.py — Static letter data collection for the Random Forest pipeline.

Captures single-frame hand landmarks for ASL fingerspelling letters A–Y
(excluding J and Z which require motion). Uses MediaPipe Hands (not Holistic).

Each letter is saved as letter_data/{LETTER}.npy with shape (N, 63).

Usage:
    python collect_letters.py
"""

import os

import cv2
import numpy as np
import mediapipe as mp


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# 24 static ASL letters (J and Z require motion).
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

OUTPUT_DIR = "letter_data"
SAMPLES_PER_LETTER = 50


# ──────────────────────────────────────────────
# Normalization (must match train_letters.py and app_simple.py)
# ──────────────────────────────────────────────

def normalize_hand_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks into a normalized 63-feature vector.

    Normalization:
        1. Extract (x, y, z) for all 21 landmarks → shape (21, 3).
        2. Shift relative to wrist (landmark 0) — translation invariance.
        3. Scale by palm size (wrist → middle MCP distance) — scale invariance.
        4. Flatten to 63 values.

    Args:
        hand_landmarks: MediaPipe hand_landmarks object.

    Returns:
        np.ndarray of shape (63,), dtype float32.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )  # (21, 3)

    # Shift relative to wrist.
    coords = coords - coords[0]

    # Scale by palm size (wrist to middle finger MCP = landmark 9).
    palm_size = np.linalg.norm(coords[9])
    if palm_size < 1e-6:
        pass  # Skip scaling for degenerate cases
    else:
        coords = coords / palm_size

    return coords.flatten().astype(np.float32)  # (63,)


# ──────────────────────────────────────────────
# Main collection function
# ──────────────────────────────────────────────

def collect_letters():
    """
    Interactive letter data collection with webcam.

    Controls:
        SPACE — Capture sample (green border flash confirms capture)
        N     — Next letter
        P     — Previous letter
        Q     — Quit and save
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize MediaPipe Hands (NOT Holistic).
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Load existing data so we can resume.
    letter_data = {}
    for letter in LETTERS:
        npy_path = os.path.join(OUTPUT_DIR, f"{letter}.npy")
        if os.path.exists(npy_path):
            letter_data[letter] = list(np.load(npy_path))
            print(f"  Loaded {len(letter_data[letter])} existing samples for '{letter}'")
        else:
            letter_data[letter] = []

    current_idx = 0
    hand_detected = False

    print("\n" + "=" * 50)
    print("  LETTER DATA COLLECTION")
    print("=" * 50)
    print("  SPACE=capture  N=next  P=prev  Q=quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        current_letter = LETTERS[current_idx]
        sample_count = len(letter_data[current_letter])
        total_collected = sum(len(v) for v in letter_data.values())
        total_needed = len(LETTERS) * SAMPLES_PER_LETTER

        # Process with MediaPipe.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # ── Top bar (semi-transparent) ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Letter and count.
        cv2.putText(
            frame,
            f"Letter: {current_letter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Samples: {sample_count}/{SAMPLES_PER_LETTER}",
            (250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Overall: {total_collected}/{total_needed}",
            (250, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # Hand detection status at bottom center.
        status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame,
            status_text,
            ((w - text_size[0]) // 2, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        # Controls instruction at bottom left.
        cv2.putText(
            frame,
            "SPACE=capture  N=next  P=prev  Q=quit",
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Letter Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──
        if key == ord(" "):
            # Capture sample.
            if hand_detected and results.multi_hand_landmarks:
                features = normalize_hand_landmarks(results.multi_hand_landmarks[0])
                letter_data[current_letter].append(features)
                sample_count = len(letter_data[current_letter])
                print(f"  Captured '{current_letter}': {sample_count}/{SAMPLES_PER_LETTER}")

                # Green border flash.
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 10)
                cv2.imshow("Letter Collection", frame)
                # cv2.waitKey(100)  # Removed slow delay for faster capture

                # Auto-advance removed (user request).
                # if sample_count >= SAMPLES_PER_LETTER:
                #     if current_idx < len(LETTERS) - 1:
                #         current_idx += 1
                #         print(f"  Auto-advancing to '{LETTERS[current_idx]}'")

        elif key == ord("n"):
            current_idx = min(current_idx + 1, len(LETTERS) - 1)
            print(f"  Switched to '{LETTERS[current_idx]}'")

        elif key == ord("p"):
            current_idx = max(current_idx - 1, 0)
            print(f"  Switched to '{LETTERS[current_idx]}'")

        elif key == ord("q"):
            break

    # ── Save all data ──
    for letter, samples in letter_data.items():
        if len(samples) > 0:
            arr = np.array(samples)  # shape (N, 63)
            np.save(os.path.join(OUTPUT_DIR, f"{letter}.npy"), arr)
            print(f"  Saved {len(samples)} samples for '{letter}'")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nDone! Now run:  python train_letters.py")


if __name__ == "__main__":
    collect_letters()

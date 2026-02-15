"""
test_camera.py — Camera & MediaPipe test utility.

Opens the webcam, runs LandmarkExtractor, and displays landmark detection
status. Useful for verifying that the camera and MediaPipe are working.

Usage:
    python test_camera.py
"""

import cv2
from LandmarkExtractor import LandmarkExtractor
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        print("  - Close other apps using the camera")
        print("  - Try a different camera index")
        return

    extractor = LandmarkExtractor()
    frame_count = 0

    print("=" * 50)
    print("  ASL CAMERA TEST")
    print("=" * 50)
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        landmarks, annotated = extractor.process_frame_with_drawing(frame, thickness=1)

        frame_count += 1

        # Check if landmarks are non-zero (i.e., pose detected).
        detected = np.any(landmarks != 0)

        # ── Overlay text ──
        cv2.putText(
            annotated,
            "ASL Landmark Detection Test",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        status_text = "DETECTED" if detected else "NONE"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(
            annotated,
            f"Landmarks: {status_text}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        cv2.putText(
            annotated,
            f"Features: {landmarks.shape[0]}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            annotated,
            f"Frame: {frame_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            annotated,
            "Press 'q' to quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

        cv2.imshow("ASL Camera Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"\nTest complete. Total frames: {frame_count}")


if __name__ == "__main__":
    main()

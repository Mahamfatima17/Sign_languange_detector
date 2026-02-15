"""
LandmarkExtractor.py — MediaPipe Holistic landmark extraction & normalization.

Used by the LSTM dynamic-sign pipeline. Extracts 1,662 features per frame
from pose, face, left hand, and right hand landmarks.

Normalization strategy:
    - Reference point: Nose (pose landmark 0)
    - Scale factor: Shoulder distance (pose landmarks 11–12)
    - All XYZ coordinates are nose-relative and shoulder-scaled
"""

import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    """
    Wraps MediaPipe Holistic to extract and normalize full-body landmarks.

    Usage:
        with LandmarkExtractor() as extractor:
            landmarks = extractor.extract_landmarks(frame)
            # landmarks.shape == (1662,)
    """

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initialize MediaPipe Holistic.

        Args:
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence:  Minimum confidence for frame-to-frame tracking.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,             # Simplified for performance
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Extract and normalize landmarks from a BGR image.

        Args:
            image: BGR image (numpy array from OpenCV).

        Returns:
            Flat numpy array of shape (1662,) with normalized features.
            Returns zeros if no pose is detected.
        """
        # Convert BGR → RGB; disable writeable flag for performance.
        image_rgb = image.copy()
        image_rgb.flags.writeable = False
        image_rgb = np.ascontiguousarray(image_rgb[:, :, ::-1])  # BGR→RGB

        results = self.holistic.process(image_rgb)

        # If no pose detected, return zeros.
        if results.pose_landmarks is None:
            return np.zeros(1662, dtype=np.float32)

        # Extract each landmark group.
        pose = self._extract_pose(results.pose_landmarks)
        face = self._extract_face(results.face_landmarks)
        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)

        # Normalize and flatten.
        return self._normalize_landmarks(pose, face, left_hand, right_hand)

    def draw_landmarks(self, image, results, thickness=1):
        """
        Draw hand landmarks on the image.

        Only draws hands (pose and face drawing are disabled for clarity).

        Args:
            image:     BGR image to draw on (modified in place).
            results:   MediaPipe Holistic results object.
            thickness: Line thickness for connections.
        """
        # Left hand — red
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(200, 0, 0), thickness=thickness, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=thickness, circle_radius=3
                ),
            )

        # Right hand — blue
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 0, 200), thickness=thickness, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=thickness, circle_radius=3
                ),
            )

    def process_frame_with_drawing(self, image, thickness=2):
        """
        Convenience method: extract landmarks AND draw on the image.

        Args:
            image:     BGR image (will be annotated in place).
            thickness: Line thickness for landmark drawing.

        Returns:
            Tuple of (landmarks, annotated_image).
            landmarks: shape (1662,) normalized feature array.
            annotated_image: the input image with landmarks drawn.
        """
        image_rgb = image.copy()
        image_rgb.flags.writeable = False
        image_rgb = np.ascontiguousarray(image_rgb[:, :, ::-1])

        results = self.holistic.process(image_rgb)

        if results.pose_landmarks is None:
            return np.zeros(1662, dtype=np.float32), image

        pose = self._extract_pose(results.pose_landmarks)
        face = self._extract_face(results.face_landmarks)
        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)

        landmarks = self._normalize_landmarks(pose, face, left_hand, right_hand)

        # Draw landmarks on the original image.
        image.flags.writeable = True
        self.draw_landmarks(image, results, thickness)

        return landmarks, image

    # ──────────────────────────────────────────
    # Private extraction methods
    # ──────────────────────────────────────────

    def _extract_pose(self, landmarks):
        """Extract pose landmarks → (33, 4) array [x, y, z, visibility]."""
        if landmarks is None:
            return np.zeros((33, 4), dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    def _extract_face(self, landmarks):
        """Extract face landmarks → (468, 3) array [x, y, z]."""
        if landmarks is None:
            return np.zeros((468, 3), dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    def _extract_hand(self, landmarks):
        """Extract hand landmarks → (21, 3) array [x, y, z]."""
        if landmarks is None:
            return np.zeros((21, 3), dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    def _normalize_landmarks(self, pose, face, left_hand, right_hand):
        """
        Normalize all landmarks for position & scale invariance.

        Strategy:
            1. Reference point  = Nose (pose[0, :3])
            2. Scale factor     = Shoulder distance (pose[11] ↔ pose[12])
            3. Subtract nose, divide by shoulder distance for all XYZ coords.
            4. Pose visibility values are kept as-is.

        Concatenation order (critical — must match training):
            left_hand  (63) + right_hand (63) + pose_xyz (99)
            + face (1404) + pose_visibility (33) = 1662

        Returns:
            np.ndarray of shape (1662,), dtype float32.
        """
        # Reference point: nose position.
        nose = pose[0, :3]

        # Scale factor: distance between left and right shoulder.
        shoulder_distance = np.linalg.norm(pose[11, :3] - pose[12, :3])
        if shoulder_distance < 1e-6:
            shoulder_distance = 1.0

        # Normalize hand coordinates.
        left_hand_normalized = (left_hand - nose) / shoulder_distance
        right_hand_normalized = (right_hand - nose) / shoulder_distance

        # Normalize pose XYZ (keep visibility separate).
        pose_xyz = pose[:, :3]
        pose_xyz_normalized = (pose_xyz - nose) / shoulder_distance
        pose_visibility = pose[:, 3]  # Not normalized

        # Normalize face coordinates.
        face_normalized = (face - nose) / shoulder_distance

        # Concatenate in the exact order used during training.
        result = np.concatenate([
            left_hand_normalized.flatten(),     # 21 * 3 =   63
            right_hand_normalized.flatten(),    # 21 * 3 =   63
            pose_xyz_normalized.flatten(),      # 33 * 3 =   99
            face_normalized.flatten(),          # 468 * 3 = 1404
            pose_visibility.flatten(),          # 33     =   33
        ])                                      # Total  = 1662

        assert result.shape == (1662,), f"Expected 1662, got {result.shape}"
        return result.astype(np.float32)

    # ──────────────────────────────────────────
    # Context manager support
    # ──────────────────────────────────────────

    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

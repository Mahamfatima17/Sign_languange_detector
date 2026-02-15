"""
collect_data.py — LSTM pipeline: Collect 30-frame sequences for a single sign.

Opens a webcam, shows a UI for the user to perform the sign, captures
30-frame landmark sequences, and saves each as a .npy file.

Usage:
    python collect_data.py
"""

import os
import time

import cv2
import numpy as np

from LandmarkExtractor import LandmarkExtractor


def collect_data_for_sign(
    sign_label,
    sign_index,
    num_sequences=30,
    sequence_length=30,
    output_dir="training_data",
):
    """
    Collect 30-frame landmark sequences for a single sign.

    Args:
        sign_label:      Name of the sign (e.g. "hello").
        sign_index:      Integer index for the sign.
        num_sequences:   How many sequences to record.
        sequence_length: Frames per sequence.
        output_dir:      Root directory for saved data.
    """
    # Create output folder: training_data/0001_hello/
    folder_name = f"{sign_index:04d}_{sign_label}"
    save_dir = os.path.join(output_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing sequences so we can resume.
    existing = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    start_seq = len(existing)

    if start_seq >= num_sequences:
        print(f"Already have {start_seq} sequences for '{sign_label}'. Skipping.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    extractor = LandmarkExtractor()

    print(f"\nCollecting data for: {sign_label}  (index {sign_index})")
    print(f"Sequences: {start_seq}/{num_sequences} already collected")
    print("Press SPACE to start recording, Q to quit.\n")

    seq_index = start_seq

    while seq_index < num_sequences:
        # ── Wait for SPACE ──
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Draw waiting UI.
            cv2.putText(
                frame,
                f'Sign: "{sign_label}"    Seq {seq_index + 1}/{num_sequences}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press SPACE to start recording",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                waiting = False
            elif key == ord("q"):
                print("Quit by user.")
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                return

        # ── 3-second countdown ──
        for countdown in [3, 2, 1]:
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                text = str(countdown)
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 4, 6
                )[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(
                    frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6,
                )
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

        # ── Record sequence ──
        sequence = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks, annotated = extractor.process_frame_with_drawing(frame)
            sequence.append(landmarks)

            # Progress bar overlay.
            progress = (frame_num + 1) / sequence_length
            bar_width = int(progress * 300)
            cv2.rectangle(annotated, (20, 20), (320, 50), (50, 50, 50), -1)
            cv2.rectangle(annotated, (20, 20), (20 + bar_width, 50), (0, 255, 0), -1)
            cv2.putText(
                annotated,
                f"Recording: {frame_num + 1}/{sequence_length}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Data Collection", annotated)
            cv2.waitKey(1)

        # Save sequence.
        seq_array = np.array(sequence)  # shape (30, 1662)
        save_path = os.path.join(save_dir, f"sequence_{seq_index:04d}.npy")
        np.save(save_path, seq_array)
        print(f"  Saved sequence {seq_index + 1}/{num_sequences} -> {save_path}")
        seq_index += 1

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"\nDone! Collected {num_sequences} sequences for '{sign_label}'.")


def load_collected_data(data_dir="training_data"):
    """
    Load all collected sequences from disk.

    Returns:
        X:           np.ndarray, shape (num_samples, 30, 1662).
        y:           np.ndarray, integer labels per sample.
        labels_dict: dict mapping int index → sign name.
    """
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory '{data_dir}' not found.")
        return None, None, None

    X_list = []
    y_list = []
    labels_dict = {}

    # Iterate sorted directories.
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse folder name: "0001_hello" → index=1, label="hello"
        parts = folder.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            sign_index = int(parts[0])
        except ValueError:
            continue
        sign_label = parts[1]

        labels_dict[sign_index] = sign_label

        # Load all .npy files in this folder.
        npy_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        )
        for npy_file in npy_files:
            seq = np.load(os.path.join(folder_path, npy_file))
            X_list.append(seq)
            y_list.append(sign_index)

    if len(X_list) == 0:
        print("No data found.")
        return None, None, None

    X = np.array(X_list)  # (num_samples, 30, 1662)
    y = np.array(y_list)

    print(f"Loaded {len(X)} sequences across {len(labels_dict)} classes.")
    print(f"X shape: {X.shape},  y shape: {y.shape}")

    return X, y, labels_dict


def main():
    """Interactive CLI for single-sign data collection."""
    print("=" * 50)
    print("  LSTM Data Collection (single sign)")
    print("=" * 50)

    sign_label = input("Enter sign name (e.g. 'hello'): ").strip()
    if not sign_label:
        print("No label entered. Exiting.")
        return

    try:
        sign_index = int(input("Enter sign index (integer, e.g. 0): ").strip())
    except ValueError:
        print("Invalid index. Exiting.")
        return

    try:
        num_seqs = int(
            input(f"Number of sequences to collect (default 30): ").strip() or "30"
        )
    except ValueError:
        num_seqs = 30

    collect_data_for_sign(sign_label, sign_index, num_sequences=num_seqs)


if __name__ == "__main__":
    main()

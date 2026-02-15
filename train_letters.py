"""
train_letters.py — Train a Random Forest classifier for static ASL letters.

Loads letter_data/*.npy, trains a Random Forest, saves the model and label map.

Usage:
    python train_letters.py
"""

import os
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_letter_data(data_dir="letter_data"):
    """
    Load letter landmark data from .npy files.

    Each file is named {LETTER}.npy and contains shape (N, 63).

    Returns:
        X:         np.ndarray, shape (total_samples, 63).
        y:         np.ndarray, integer labels.
        label_map: dict mapping int index → letter string.
    """
    if not os.path.exists(data_dir):
        print(f"ERROR: '{data_dir}' not found. Run collect_letters.py first.")
        return None, None, None

    npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    if not npy_files:
        print("ERROR: No .npy files found in letter_data/.")
        return None, None, None

    X_list = []
    y_list = []
    label_map = {}

    for idx, filename in enumerate(npy_files):
        letter = filename.replace(".npy", "")
        label_map[idx] = letter

        data = np.load(os.path.join(data_dir, filename))
        X_list.append(data)
        y_list.extend([idx] * len(data))

        print(f"  Loaded {len(data):>4} samples for '{letter}' (class {idx})")

    X = np.concatenate(X_list, axis=0)  # (total, 63)
    y = np.array(y_list)

    print(f"\nTotal: {len(X)} samples, {len(label_map)} classes")
    return X, y, label_map


def train():
    """Train the Random Forest and save model + labels."""

    print("=" * 50)
    print("  LETTER MODEL TRAINING")
    print("=" * 50)

    # 1. Load data.
    X, y, label_map = load_letter_data()
    if X is None:
        return

    # 2. Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    # 3. Train model.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest ...")
    model.fit(X_train, y_train)
    print("Training complete!")

    # 4. Evaluate.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%\n")

    # Build target_names in order of class indices.
    target_names = [label_map[i] for i in range(len(label_map))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. Save model.
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "letter_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    # 6. Save label map.
    labels_path = "letter_labels.json"
    json_map = {str(k): v for k, v in label_map.items()}
    with open(labels_path, "w") as f:
        json.dump(json_map, f, indent=2)
    print(f"Label map saved to {labels_path}")

    print("\nNext step -> streamlit run app_simple.py")


if __name__ == "__main__":
    train()

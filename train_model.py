"""
train_model.py — Train the LSTM model for dynamic sign recognition.

Loads collected 30-frame sequences, one-hot encodes labels, trains the
LSTM model, evaluates, and saves.

Usage:
    python train_model.py
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from collect_data import load_collected_data
from SignModel import build_model, save_model


# ──────────────────────────────────────────────
# Training history plot
# ──────────────────────────────────────────────

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot accuracy and loss curves and save to disk.

    Args:
        history:   Keras History object from model.fit().
        save_path: Where to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train")
    if "val_accuracy" in history.history:
        ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss
    ax2.plot(history.history["loss"], label="Train")
    if "val_loss" in history.history:
        ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_path}")


# ──────────────────────────────────────────────
# Main training script
# ──────────────────────────────────────────────

def main():
    # Configuration.
    DATA_DIR = "training_data"
    MODEL_SAVE_PATH = "models/asl_model.h5"
    LABELS_SAVE_PATH = "labels.json"
    EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    # 1. Load data.
    X, y, labels_dict = load_collected_data(DATA_DIR)
    if X is None:
        print("No training data found. Run collect_data.py or batch_collect_data.py first.")
        return

    num_classes = len(labels_dict)
    print(f"\nClasses: {num_classes}")
    print(f"Labels: {labels_dict}")

    # 2. One-hot encode labels.
    y_encoded = to_categorical(y, num_classes=num_classes)

    # 3. Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y,
    )

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    # 4. Build model.
    model = build_model(num_classes=num_classes)
    model.summary()

    # 5. Train.
    os.makedirs("models", exist_ok=True)
    from SignModel import train_model as sm_train
    history = sm_train(
        model, X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=MODEL_SAVE_PATH,
    )

    # 6. Plot training history.
    plot_training_history(history)

    # 7. Save labels.
    labels_json = {str(k): v for k, v in labels_dict.items()}
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(labels_json, f, indent=2)
    print(f"Labels saved to {LABELS_SAVE_PATH}")

    # 8. Final evaluation.
    print("\n" + "=" * 50)
    print("  FINAL EVALUATION")
    print("=" * 50)
    loss, accuracy, top_k = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss:          {loss:.4f}")
    print(f"  Accuracy:      {accuracy * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top_k * 100:.2f}%")


if __name__ == "__main__":
    main()

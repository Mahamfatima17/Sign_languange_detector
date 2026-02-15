"""
SignModel.py — LSTM model definition, training, prediction, and stabilizer.

Used by the dynamic-sign (word/phrase) pipeline.
Architecture: Input(30,1662) → LSTM(64) → LSTM(128) → Dense(64) → Softmax
"""

import json
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────

def build_model(num_classes, sequence_length=30, num_features=1662):
    """
    Build the LSTM model for dynamic sign recognition.

    Args:
        num_classes:     Number of sign classes.
        sequence_length: Number of frames per sequence (default 30).
        num_features:    Number of features per frame (default 1662).

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential(name="ASL_LSTM_Model")

    model.add(Input(shape=(sequence_length, num_features)))

    model.add(LSTM(
        64,
        return_sequences=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="lstm_1",
    ))
    model.add(Dropout(0.2))

    model.add(LSTM(
        128,
        return_sequences=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="lstm_2",
    ))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu", name="dense_1"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )

    return model


# ──────────────────────────────────────────────
# Model I/O
# ──────────────────────────────────────────────

def load_model(model_path):
    """Load a saved Keras model from disk."""
    return keras_load_model(model_path)


def save_model(model, save_path):
    """Save a Keras model to disk."""
    model.save(save_path)
    print(f"Model saved to {save_path}")


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=100,
    batch_size=32,
    save_path="models/asl_model.h5",
    early_stopping_patience=15,
    reduce_lr_patience=7,
):
    """
    Train the LSTM model with best-practice callbacks.

    Args:
        model:         Compiled Keras model.
        X_train:       Training features, shape (N, 30, 1662).
        y_train:       Training labels, one-hot encoded.
        X_val:         Optional validation features.
        y_val:         Optional validation labels.
        epochs:        Maximum number of training epochs.
        batch_size:    Samples per gradient update.
        save_path:     Where to save the best model.
        early_stopping_patience: Epochs with no improvement before stopping.
        reduce_lr_patience:      Epochs before reducing learning rate.

    Returns:
        Keras History object.
    """
    # Choose monitoring metric based on whether validation data exists.
    has_validation = X_val is not None and y_val is not None
    monitor_acc = "val_accuracy" if has_validation else "accuracy"
    monitor_loss = "val_loss" if has_validation else "loss"

    callbacks = [
        ModelCheckpoint(
            save_path,
            monitor=monitor_acc,
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor_loss,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor_loss,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("=" * 50)
    print("  TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples:    {len(X_train)} train"
          + (f", {len(X_val)} val" if has_validation else ""))
    print(f"  Classes:    {y_train.shape[1]}")
    print(f"  Save path:  {save_path}")
    print("=" * 50)

    validation_data = (X_val, y_val) if has_validation else None

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ──────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────

def load_labels(labels_path="labels.json"):
    """
    Load label dictionary from JSON.

    The JSON file maps string indices ("0", "1", …) to sign names.
    We convert keys to integers for easy lookup.

    Returns:
        dict: {int_index: "sign_name", ...}
    """
    with open(labels_path, "r") as f:
        labels = json.load(f)
    return {int(k): v for k, v in labels.items()}


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict_sign(model, sequence, labels, top_k=5):
    """
    Predict the sign from a 30-frame sequence.

    Args:
        model:    Trained Keras model.
        sequence: numpy array, shape (30, 1662) or (1, 30, 1662).
        labels:   dict mapping class index → sign name.
        top_k:    Number of top predictions to return.

    Returns:
        List of (label, confidence) tuples, sorted by confidence descending.
    """
    # Add batch dimension if needed.
    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)

    probabilities = model.predict(sequence, verbose=0)[0]

    # Get top-K indices sorted by probability.
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        label = labels.get(idx, f"unknown_{idx}")
        confidence = float(probabilities[idx])
        results.append((label, confidence))

    return results


# ──────────────────────────────────────────────
# Prediction stabilizer
# ──────────────────────────────────────────────

class PredictionStabilizer:
    """
    Smooths noisy frame-by-frame predictions.

    A prediction is only emitted when the same label appears in ALL
    of the last `stability_frames` frames with confidence above the
    threshold.
    """

    def __init__(self, min_confidence=0.90, stability_frames=10):
        """
        Args:
            min_confidence:   Ignore predictions below this confidence.
            stability_frames: Number of consecutive agreeing frames required.
        """
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames
        self.buffer = []

    def add_prediction(self, label, confidence):
        """
        Add a new prediction and check for stability.

        Args:
            label:      Predicted sign name.
            confidence: Model confidence (0–1).

        Returns:
            The stable label string if consensus is reached, else None.
        """
        if confidence >= self.min_confidence:
            self.buffer.append(label)
        else:
            self.buffer.append(None)

        # Keep only the last N frames.
        self.buffer = self.buffer[-self.stability_frames:]

        # Check if all N frames agree on the same non-None label.
        if len(self.buffer) == self.stability_frames:
            if self.buffer[0] is not None and len(set(self.buffer)) == 1:
                stable_label = self.buffer[0]
                self.reset()
                return stable_label

        return None

    def reset(self):
        """Clear the prediction buffer."""
        self.buffer = []


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(num_classes=100)
    model.summary()

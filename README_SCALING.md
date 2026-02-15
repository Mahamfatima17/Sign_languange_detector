# 📈 Scaling to 100+ Signs

Guide for expanding the ASL Translation System beyond the initial vocabulary.

---

## 1. Expanding the Vocabulary

### Letter Pipeline (Random Forest)

The letter pipeline already supports 24 static letters. To add **J** or **Z** (motion-based), you would need to switch to the LSTM pipeline for those specific letters or use a motion-detection heuristic.

### LSTM Pipeline

The `labels.json` file already defines 100 sign classes. To add more:

1. Edit `labels.json` — add new entries (e.g. `"100": "water", "101": "food"`)
2. Update `VOCABULARY` in `batch_collect_data.py`
3. Collect data: `python batch_collect_data.py`
4. Retrain: `python train_model.py`

---

## 2. Data Collection at Scale

### Recommended Samples

| Scale | Sequences per Sign | Training Time |
|-------|-------------------|---------------|
| Prototype (5–10 signs) | 30 | Minutes |
| Small (20–30 signs) | 50 | ~30 min |
| Medium (50–100 signs) | 100 | ~2 hours |
| Large (100+ signs) | 200+ | Several hours |

### Tips for Quality Data

- **Consistent lighting** — Avoid shadows and backlight
- **Multiple angles** — Slightly vary hand position between sequences
- **Different backgrounds** — Helps the model generalize
- **Both hands** — Some signs use both hands; ensure both are visible
- **Multiple signers** — If possible, collect from 2–3 different people

### Batch Collection

Use `batch_collect_data.py` with the `continue_previous_collection()` option to resume interrupted sessions. Your progress is saved automatically.

---

## 3. Model Architecture Scaling

### When to Scale the LSTM

| Signs | Architecture | Notes |
|-------|-------------|-------|
| <30 | Default (64 → 128) | Works well |
| 30–100 | Increase to (128 → 256) | Edit `SignModel.py` |
| 100+ | Add a third LSTM layer | More complex temporal patterns |

### How to Scale

Edit `build_model()` in `SignModel.py`:

```python
# For 100+ signs:
model.add(LSTM(128, return_sequences=True, name="lstm_1"))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True, name="lstm_2"))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False, name="lstm_3"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu", name="dense_1"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax", name="output"))
```

### Random Forest Scaling

For >24 letter classes, increase:
```python
RandomForestClassifier(
    n_estimators=500,      # More trees
    max_depth=30,          # Deeper trees
    min_samples_split=3,
    min_samples_leaf=1,
    n_jobs=-1,
)
```

---

## 4. Handling Class Imbalance

If some signs have more training data than others:

### Option A: Data Augmentation
- Add slight noise to landmark coordinates
- Mirror hand data (left ↔ right) where the sign is symmetric

### Option B: Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = dict(zip(np.unique(y), class_weights))

# Pass to model.fit()
model.fit(X_train, y_train, class_weight=weight_dict, ...)
```

### Option C: Oversampling
```python
from sklearn.utils import resample

# Upsample minority classes to match the largest class
```

---

## 5. Performance Optimization

| Technique | Impact | Complexity |
|-----------|--------|-----------|
| Reduce `model_complexity` to 0 | Faster inference | Low |
| Use `static_image_mode=False` | Faster tracking | Low |
| Skip frames (process every 2nd) | 2× faster | Low |
| Use GPU (TensorFlow-GPU) | 5–10× faster training | Medium |
| Quantize model (TFLite) | Smaller, faster inference | High |

---

## 6. Deployment Tips

- **Web deployment:** Use Streamlit Cloud (free tier) with `streamlit-webrtc`
- **Edge deployment:** Convert to TFLite for mobile / Raspberry Pi
- **API deployment:** Wrap inference in Flask / FastAPI for backend use
- **Desktop app:** Package with PyInstaller for standalone executable

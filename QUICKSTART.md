# ⚡ Quick Start Guide

Get the ASL letter detector running in **5 minutes**.

---

## 1. Install

```bash
# Create Python 3.9 venv
python3.9 -m venv .venv39

# Activate
# Windows:
.venv39\Scripts\activate
# macOS/Linux:
source .venv39/bin/activate

# Install deps
pip install -r requirements.txt
```

## 2. Collect Letter Data

```bash
python collect_letters.py
```

- **SPACE** = capture a sample
- **N** / **P** = next / previous letter
- **Q** = quit and save

Collect ~50 samples per letter for best results.

## 3. Train the Model

```bash
python train_letters.py
```

Expect **90%+** accuracy with clean data.

## 4. Run the App

```bash
streamlit run app_simple.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

Click **▶️ Start Camera** to begin detecting letters!

---

## Tips

- Keep your hand clearly visible and centered
- Good lighting improves detection accuracy
- If the camera doesn't work, try changing the Camera index in the sidebar
- To add more letters: edit `LETTERS` in `collect_letters.py`, re-collect, re-train

## Next Steps

- See [README.md](README.md) for the full documentation
- See [README_SCALING.md](README_SCALING.md) for scaling to 100+ signs
- Try the LSTM pipeline for word/phrase detection: `python batch_collect_data.py`

# Sign_languange_detector
The ASL Translation System is a real-time computer vision application designed to interpret American Sign Language (ASL) using a standard webcam. It features two distinct detection pipelines: one for static letters (fingerspelling) and another for dynamic words and phrases.
# 🤟 ASL Translation System

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![MediaPipe 0.10.9](https://img.shields.io/badge/MediaPipe-0.10.9-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Mahamfatima17/Sign_languange_detector)
![License MIT](https://img.shields.io/badge/License-MIT-yellow)

Real-time American Sign Language detection system using computer vision and deep learning. This project features two independent pipelines for recognizing static fingerspelling (letters) and dynamic signs (words/phrases).

## 🚀 Key Capabilities

- **🔤 Static Letter Detection**: Real-time recognition of ASL alphabet (A–Y, excluding J/Z) using MediaPipe Hands and Random Forest.
- **🖐️ Dynamic Sign Detection**: Sequence recognition for words and phrases using MediaPipe Holistic and LSTM networks.
- **📹 Live Inference**: Interactive webcam feed with landmark visualization.
- **📊 Confidence Scoring**: Real-time prediction probabilities and stabilization.
- **🔄 End-to-End Workflow**: Complete scripts for data collection, training, and inference.

---

## 🛠️ Technologies Used

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Hand Tracking** | MediaPipe Hands | Extracts 21 3D landmarks per hand |
| **Pose Estimator** | MediaPipe Holistic | Full-body tracking for dynamic signs |
| **Letter Classifier** | Scikit-learn | Random Forest for static poses |
| **Sign Classifier** | TensorFlow/Keras | LSTM for temporal sequence analysis |
| **UI Framework** | Streamlit | Web-based interface for real-time visualization |
| **Image Processing** | OpenCV | Frame capture and drawing utilities |

---

## 📂 Project Structure

```
sign-language-detector/
├── app.py                      # Main Streamlit app (Dynamic Signs + HUD)
├── app_simple.py               # Lightweight Streamlit app (Static Letters)
├── collect_letters.py          # Data collection for static letters
├── train_letters.py            # Training script for letter model (Random Forest)
├── batch_collect_data.py       # Data collection for dynamic signs
├── train_model.py              # Training script for sign model (LSTM)
├── LandmarkExtractor.py        # Helper for MediaPipe Holistic extraction
├── SignModel.py                # LSTM model definition and utilities
├── requirements.txt            # Python dependencies
├── run_app.sh                  # Helper script to launch the app
├── README.md                   # Project documentation
├── QUICKSTART.md               # Quick start guide
└── models/                     # Directory for trained models
    ├── letter_model.pkl        # Trained Random Forest model
    └── asl_model.h5            # Trained LSTM model
```

---

## ⚡ Installation

### Prerequisites
- Python 3.9 (Recommended for MediaPipe compatibility)
- Webcam

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mahamfatima17/Sign_languange_detector.git
   cd Sign_languange_detector
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### Option A: Static Letter Detection (Quick Start)

1. **Collect Data** (Optional if using pre-trained model)
   ```bash
   python collect_letters.py
   ```
   *Press `SPACE` to capture, `N`/`P` to change letters, `Q` to quit.*

2. **Train Model**
   ```bash
   python train_letters.py
   ```

3. **Run App**
   ```bash
   streamlit run app_simple.py
   ```

### Option B: Dynamic Sign Detection (Advanced)

1. **Collect Data**
   ```bash
   python batch_collect_data.py
   ```
   *Follow on-screen prompts to record 30 frames per sign.*

2. **Train Model**
   ```bash
   python train_model.py
   ```

3. **Run App**
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Model Details

### 1. Static Letter Model
- **Input**: 63 normalized hand landmark coordinates (x, y, z).
- **Algorithm**: Random Forest Classifier.
- **Performance**: High accuracy (~95%+) on static poses with minimal latency.

### 2. Dynamic Sign Model
- **Input**: Sequence of 30 frames × 1662 landmarks (Pose + Face + Hands).
- **Architecture**:
  - `LSTM` (64 units) → `Dropout`
  - `LSTM` (128 units) → `Dropout`
  - `Dense` (64 units, ReLU)
  - `Dense` (Softmax)
- **Features**: Translates temporal movement patterns into sign predictions.

---

## 🔧 Troubleshooting

- **MediaPipe Errors**: Ensure you use `mediapipe==0.10.9`. Newer versions may have API changes.
- **Camera Issues**: If the camera doesn't open, try changing the `camera_index` in `app.py` or the sidebar settings.
- **Low Accuracy**: Ensure good lighting and keep your hand within the frame. Re-collecting data with your own hand usually improves performance significantly.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Acknowledgments

- Built using [MediaPipe](https://mediapipe.dev/) and [Streamlit](https://streamlit.io/).
- Inspired by various open-source ASL recognition projects.

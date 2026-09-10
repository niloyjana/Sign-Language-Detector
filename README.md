# Sign Language Detector — Real-Time Hand Gesture Recognition

Sign Language Detector is a real-time hand-gesture recognition system built with OpenCV, MediaPipe, and scikit-learn. It tracks your hand through a webcam, extracts hand-landmark keypoints, and classifies the gesture using a trained SVM model, displaying the prediction and confidence live on screen.

---

## Key Features

- **Hand Landmark Tracking**: Uses MediaPipe Hands to detect 21 keypoints per hand in real time, for up to 2 hands simultaneously.
- **Custom Dataset Collection**: Built-in webcam tool to capture and label your own hand-sign images for any gesture set.
- **SVM Classification**: Trains a linear-kernel SVM on extracted landmark coordinates for lightweight, fast inference.
- **Live Inference Overlay**: Displays a bounding box, hand side (Left/Right), predicted gesture, and confidence score directly on the video feed.

---

## Project Structure

```text
Sign-Language-Detector/
├── collect_imgs.py       # Step 1: capture labeled hand-sign images from webcam
├── create_dataset.py     # Step 2: extract MediaPipe landmarks into dataset.p
├── train_classifier.py   # Step 3: train an SVM on dataset.p, save model.p
├── run_inference.py      # Step 4: real-time webcam gesture recognition
├── dataset.p             # Pickled (landmarks, labels) dataset
├── model.p               # Pickled trained SVM model
└── requirements.txt      # System dependencies
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/niloyjana/Sign-Language-Detector.git
cd Sign-Language-Detector
```

### 2. Environment Setup
We recommend using a virtual environment (Python 3.8–3.10, for MediaPipe compatibility):
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Recognition Pipeline

1. **Image Collection**: `collect_imgs.py` prompts for a label and opens your webcam — press S to save a frame, Q to quit. Images are saved to `data/<label>/`.
2. **Landmark Extraction**: `create_dataset.py` runs MediaPipe Hands over every collected image and stores the (x, y) coordinates of all 21 landmarks per sample in `dataset.p`.
3. **Model Training**: `train_classifier.py` splits the dataset 80/20, trains a linear-kernel SVM, prints test accuracy, and saves the model to `model.p`.
4. **Real-Time Inference**: `run_inference.py` opens the webcam, tracks hands live, and overlays the predicted gesture with confidence and hand side for each detected hand.

---

## Custom Training

You can train the detector on your own sign set:
1. Run `collect_imgs.py` once per label you want to teach the model.
2. Run:
   ```bash
   python create_dataset.py
   python train_classifier.py
   ```
3. Run `python run_inference.py` to test live.

---

## Requirements

```text
opencv-python==4.8.1.78
mediapipe==0.10.9
numpy==1.24.4
scikit-learn==1.3.2
protobuf<4
pandas
```

---

Developed as a computer vision project for real-time sign language recognition.

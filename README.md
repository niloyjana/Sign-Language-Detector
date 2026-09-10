# 🤟 Sign Language Detector

A real-time hand-gesture / sign-language recognizer built with **OpenCV**, **MediaPipe**, and **scikit-learn**. It tracks your hand through a webcam, extracts hand-landmark keypoints, and classifies the gesture using a trained SVM model — showing the prediction and confidence live on screen.

## 🚀 How It Works

The pipeline has three stages:

1. **Collect images** — capture labeled photos of hand signs from your webcam.
2. **Build a dataset** — extract 2D hand-landmark keypoints (via MediaPipe) from each image and save them as a pickled dataset.
3. **Train & run** — train an SVM classifier on the keypoints, then run it live for real-time gesture recognition, complete with a bounding box, left/right hand label, and confidence score overlay.

## 🛠️ Tech Stack

- **OpenCV** – webcam capture & UI rendering
- **MediaPipe Hands** – hand landmark detection (21 keypoints per hand)
- **scikit-learn (SVM)** – gesture classification
- **NumPy / Pandas** – data handling

## 📂 Project Structure

```text
Sign-Language-Detector/
├── collect_imgs.py       # Step 1: capture labeled hand-sign images from webcam
├── create_dataset.py     # Step 2: extract MediaPipe landmarks → dataset.p
├── train_classifier.py   # Step 3: train an SVM on dataset.p → model.p
├── run_inference.py      # Step 4: real-time webcam gesture recognition
├── dataset.p             # Pickled (landmarks, labels) dataset
├── model.p               # Pickled trained SVM model
└── requirements.txt
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.8–3.10 (MediaPipe compatibility)
- A webcam

### Installation

```bash
git clone https://github.com/niloyjana/Sign-Language-Detector.git
cd Sign-Language-Detector
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

**1. Collect training images**

Run this once per sign/label you want to teach the model:

```bash
python collect_imgs.py
```

You'll be prompted for a label (e.g. `A`). A webcam window opens — press **S** to save a frame, **Q** to quit. Images are saved to `data/<label>/`.

**2. Build the dataset**

Extracts hand-landmark keypoints from every collected image:

```bash
python create_dataset.py
```

This produces `dataset.p`.

**3. Train the classifier**

```bash
python train_classifier.py
```

Trains an SVM (linear kernel) on an 80/20 train-test split, prints accuracy, and saves `model.p`.

**4. Run real-time detection**

```bash
python run_inference.py
```

Opens your webcam, tracks up to 2 hands, and overlays the predicted gesture with confidence and hand side (Left/Right). Press **Esc** to quit.

## 📦 Requirements

```text
opencv-python==4.8.1.78
mediapipe==0.10.9
numpy==1.24.4
scikit-learn==1.3.2
protobuf<4
pandas
```

## 📝 Notes

- `dataset.p` and `model.p` are already included in the repo (pre-trained), but you can regenerate them by re-running steps 1–3 with your own signs/labels.
- Detection confidence and tracking confidence thresholds can be tuned in `run_inference.py` (`min_detection_confidence`, `min_tracking_confidence`).

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

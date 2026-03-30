# 🖐️ Real-Time Sign Language & Gesture Detector

A professional-grade real-time hand gesture recognition system built with **Python**, **OpenCV**, **MediaPipe**, and **Scikit-Learn**. 

This project tracks 21 hand landmarks, detects handedness (Left/Right), and classifies gestures with high precision using a Support Vector Classifier (SVC).

---

## ✨ Key Features

- **🎯 Precision Tracking**: Real-time tracking of 21 hand landmarks using MediaPipe.
- **🤝 Multi-Hand Support**: Detect and classify up to **two hands** simultaneously.
- **🌗 Handedness Detection**: Automatically distinguishes between Left and Right hands.
- **🏷️ Real-Time Labeling**: Displays gesture names and prediction confidence (0-100%) in a dynamic UI.
- **🔲 Follow-Hand UI**: Padded bounding boxes and skeletal overlays that follow hand movement.
- **🪞 Mirror Mode**: Horizontally flipped camera feed for a natural user experience.
- **📦 Custom Dataset Suite**: Full pipeline included to collect, process, and train your own gestures.

---

## 🛠️ Technology Stack

- **Python 3.9+**
- **MediaPipe**: For high-performance hand landmark detection.
- **OpenCV**: For camera feed processing and UI rendering.
- **Scikit-Learn**: For the machine learning classification (SVC).
- **NumPy & Pickle**: For data manipulation and model serialization.

---

## 📸 Collecting Your Own Dataset

You can easily train the system to recognize **any** custom gesture by collecting your own data. Follow these steps:

### Step 1: Capture Images
Run the collection script to grab images of your custom gesture:
```bash
python collect_imgs.py
```
- **Enter a Label**: (e.g., 'A', 'B', 'Happy', 'Peace')
- **Save Images**: Press **'S'** while making the gesture to capture frames. Aim for 80-100 images per gesture for better accuracy.
- **Quit**: Press **'Q'** once you've captured enough data.

### Step 2: Create the Dataset
Extract hand features from your captured images and prepare the data for training:
```bash
python create_dataset.py
```
This will generate a `dataset.p` file containing the normalized (x, y) coordinates for all images in the `data/` folder.

### Step 3: Train the Model
Train the SVC classifier using your newly created dataset:
```bash
python train_classifier.py
```
This will report the final model accuracy and save the trained weights as `model.p`.

---

## ⏯️ Running the Detector

Once your model is trained (or using the included `model.p`), start the real-time inference:

```bash
python run_inference.py
```

- **Controls**: Press **ESC** to close the camera window.
- **UI Details**: The top of the bounding box shows `Hand Side | Gesture (Confidence%)`.

---

## 📁 Project Structure

```text
sign-language-detector/
├── data/               # Raw gesture images (organized by label)
├── dataset.p           # Processed landmark data
├── model.p             # Trained SVC model
├── collect_imgs.py     # Script to capture your own data
├── create_dataset.py   # Script to extract features from images
├── train_classifier.py # Script to train the ML model
├── run_inference.py    # Main script for real-time detection
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---
## 🎥 Demo Video
Watch the project in action below 👇


https://github.com/user-attachments/assets/8a9d09db-ca53-4e57-93b2-0bb34aa26a66

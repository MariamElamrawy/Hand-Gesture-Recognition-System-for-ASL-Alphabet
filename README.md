Hand Gesture Recognition System for ASL Alphabet

American Sign Language (A–Z) Classification Using Computer Vision & CNNs

This project implements a real-time ASL alphabet recognition system using a laptop webcam, OpenCV, MediaPipe/cvzone for hand detection, and a custom-trained Convolutional Neural Network (CNN). The system captures hand images, preprocesses them, classifies the gesture, and outputs the predicted ASL letter in real time.

Overview

This system recognizes static ASL alphabet gestures (A–Z) from webcam images.
It includes:
A fully custom dataset of 5,000+ labeled hand gesture images
A preprocessing pipeline using OpenCV and cvzone HandDetector
A custom 128×128 CNN trained in TensorFlow/Keras
Real-time gesture classification using webcam input
Confusion matrix, accuracy/loss visualization, and error analysis

The goal is to support accessible communication between ASL users and non-signers using only a standard laptop webcam.

AI Methods Used:

Hand Detection & Preprocessing

cvzone HandDetector (MediaPipe Hands backend)
Bounding box extraction of the dominant hand
Aspect-ratio–preserving resize with centered padding
Standardized white background (128×128 px)
Normalization to [0,1] pixel scaling

Model Architecture

A custom TensorFlow/Keras CNN:
3× Conv2D + BatchNorm + MaxPooling blocks
Dropout (0.2 → 0.5) to reduce overfitting
Dense(256) classifier + softmax output
Trained using Adam optimizer (LR = 0.0005)

Project Structure

Hand-Gesture-Recognition-System-for-ASL-Alphabet/
|--- datacollection.py
|--- split_dataset.py
|--- train.py
|--- confusion_matrix.py
|--- test.py
|--- test_accuracy.py
|--- requirements.txt
|--- README.md
|---converted_keras/
  |--- keras_model.h5 (pre-trained model)
  |--- labels.txt
|--- custom_dataset (pre-collected images)
|--- split/ (pre-split data)
  |--- test
  |--- train
  |--- val

Getting Started (VS Code setup)

After downloading the files, open the VS Code terminal. In the VS Code terminal, create a virtual environment and activate it:

python -m venv .venv
                               source .venv/bin/activate      #macOS/Linux
                            .venv\Scripts\activate          # Windows

Then open the command palette and click on Python: Select Interpreter and choose:

./.venv/bin/python

to ensure VS Code runs the correct Python environment.

After creating the environment and downloading the dependencies, run the files in order:

datacollection.py 
split_dataset.py
train.py
confusion_matrix.py (optional)
test_accuracy.py
test.py (real-time webcam demo)

Requirements File

This project includes a requirements.txt file that lists all Python dependencies needed to run the ASL recognition system on both macOS and Windows/Linux. The file provides OS-specific installation instructions to ensure full compatibility. This helps guarantee that your environment matches ours so the model, training scripts, and real-time webcam inference run smoothly. To avoid TensorFlow compatibility issues, this project requires Python 3.11.

Note: macOS users must install `tensorflow-macos` and `tensorflow-metal`, while Windows/Linux users must install standard `tensorflow`. Instructions for both are included in requirements.txt.
For windows: If the program is unable to recognize tensorflow, please reinstall it (pip install tensorflow) and close & open the folder again. 

To install everything at once, run

pip install -r requirements.txt

This installs:
TensorFlow 2.16.x - for training and loading the CNN (IMPORTANT)
OpenCV - for image capture and preprocessing
MediaPipe - for hand detection and landmarks
cvzone - wrapper around MediaPipe for easier bounding boxes
NumPy 1.26.x - confusion matrix operations (NumPy 2.0 breaks TensorFlow)
Matplotlib - visualization of training curves and confusion matrix
Scikit-learn - evaluation metrics


Data Collection File

Since our team created a fully custom ASL dataset, the datacollection.py script is included only for reference in case you want to capture your own images. You do not need to run this file, because the complete dataset we used is already provided in the project ZIP.
If using the provided dataset, skip directly to:
python split_dataset.py
If you prefer to skip training and use the pre-trained model directly, simply run:
python test.py






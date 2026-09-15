# ✍️ Handwritten Character Recognition

A deep learning project that recognizes handwritten digits from images or a drawing canvas using a **Convolutional Neural Network (CNN)**. The project includes a user-friendly **Tkinter GUI** that allows users to draw or upload a handwritten digit and receive a predicted result.

## 📌 Project Overview

Handwritten Character Recognition is an application of Artificial Intelligence and Computer Vision that converts handwritten characters into machine-readable information.

In this project, a CNN model is trained to recognize handwritten digits. Users can either **draw a digit on the canvas** or **upload an image**, and the application processes the input and predicts the digit.

## 🎯 Objectives

* Recognize handwritten digits using Deep Learning.
* Train a CNN model for image classification.
* Process handwritten images using Python.
* Provide a simple graphical user interface.
* Allow users to draw digits directly on a canvas.
* Allow users to upload handwritten digit images.
* Display the predicted digit to the user.

## 🧠 Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pillow (PIL)**
* **Tkinter**
* **Matplotlib**
* **CNN (Convolutional Neural Network)**

## 🔄 How the System Works

```text
Handwritten Digit
       ↓
Draw on Canvas / Upload Image
       ↓
Image Preprocessing
       ↓
Resize to 28 × 28
       ↓
Normalize Pixel Values
       ↓
CNN Model
       ↓
Prediction
       ↓
Display Recognized Digit
```

## 📂 Project Structure

```text
Handwritten_Character_Recognition/
│
├── gui.py
├── predict.py
├── handwritten_digit_model.keras
├── requirements.txt
├── README.md
│
├── test_images/
│   └── test_digit.png
│
└── screenshots/
    └── gui.png
```

> The exact folder and file names may vary depending on the final project setup.

## 🧩 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** for handwritten digit classification.

The model contains layers such as:

```text
Input Image
    ↓
Conv2D
    ↓
MaxPooling2D
    ↓
Conv2D
```

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageGrab
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("handwritten_digit_model.keras")

# Create window
window = tk.Tk()
window.title("Handwritten Digit Recognition")
window.geometry("500x700")


# Title
title = tk.Label(
    window,
    text="Handwritten Digit Recognition",
    font=("Arial", 20)
)
title.pack(pady=15)


# Drawing canvas
canvas = tk.Canvas(
    window,
    width=280,
    height=280,
    bg="black"
)
canvas.pack(pady=15)


# Draw on canvas
def draw(event):
    x = event.x
    y = event.y

    canvas.create_oval(
        x - 8,
        y - 8,
        x + 8,
        y + 8,
        fill="white",
        outline="white"
    )


canvas.bind("<B1-Motion>", draw)


# -----------------------------
# Predict drawn digit
# -----------------------------
# -----------------------------
# Predict drawn digit
# -----------------------------
def predict_digit():

    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()

    x2 = x + canvas.winfo_width()
    y2 = y + canvas.winfo_height()

    image = ImageGrab.grab(
        bbox=(x, y, x2, y2)
    )

    image = image.convert("L")

    image_array = np.array(image)

    # Find the handwritten digit
    coords = np.argwhere(image_array > 30)

    if coords.size == 0:
        result_label.config(
            text="Please draw a digit first"
        )
        return

    # Find bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop the digit
    cropped = image_array[y0:y1, x0:x1]

    # Make the image square
    h, w = cropped.shape
    size = max(h, w) + 20

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Center the digit
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped

    # Resize to 28x28
    image = Image.fromarray(square)

    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # Normalize
    image_array = np.array(
        image
    ).astype("float32") / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(
        1, 28, 28, 1
    )

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    digit = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    # Display result
    result_label.config(
        text=f"Prediction: {digit}\n"
             f"Confidence: {confidence:.2f}%"
    )

    # Show probability chart
    show_probability_chart(prediction)


# -----------------------------
# Upload image
# -----------------------------
def upload_image():
    
    file_path = filedialog.askopenfilename(
        title="Select Handwritten Digit Image",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        # Open image
        image = Image.open(file_path)

        # Convert to grayscale
        image = image.convert("L")

        # Resize
        image = image.resize((28, 28))

        # Convert to NumPy array
        image_array = np.array(image)

        # Normalize
        image_array = image_array.astype("float32") / 255.0

        # Reshape
        image_array = image_array.reshape(
            1, 28, 28, 1
        )

        # Predict
        prediction = model.predict(
            image_array,
            verbose=0
        )

        digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        # Display result
        result_label.config(
            text=f"Uploaded Image\n"
                 f"Prediction: {digit}\n"
                 f"Confidence: {confidence:.2f}%"
        )
        

    except Exception as error:

        result_label.config(
            text=f"Error:\n{error}"
        )
# -----------------------------
# Probability Chart
# -----------------------------
def show_probability_chart(prediction):

    probabilities = prediction[0] * 100

    digits = list(range(10))

    plt.figure(figsize=(8, 5))

    plt.bar(digits, probabilities)

    plt.xlabel("Digit")
    plt.ylabel("Probability (%)")
    plt.title("Digit Prediction Probability")

    plt.xticks(digits)

    plt.ylim(0, 100)

    plt.show()


# -----------------------------
# Clear canvas
# -----------------------------
def clear_canvas():

    canvas.delete("all")

    result_label.config(
        text="Prediction: -"
    )        
# -----------------------------
# Buttons
# -----------------------------

predict_button = tk.Button(
    window,
    text="Predict Drawing",
    command=predict_digit,
    font=("Arial", 13)
)

predict_button.pack(pady=5)


clear_button = tk.Button(
    window,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 13)
)

clear_button.pack(pady=5)


upload_button = tk.Button(
    window,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 13)
)

upload_button.pack(pady=5)


# Result
result_label = tk.Label(
    window,
    text="Prediction: -",
    font=("Arial", 18)
)

result_label.pack(pady=20)
def show_probability_chart(prediction):

    probabilities = prediction[0] * 100

    digits = list(range(10))

    plt.figure(figsize=(8, 5))

    plt.bar(digits, probabilities)

    plt.xlabel("Digit")
    plt.ylabel("Probability (%)")
    plt.title("Digit Prediction Probability")

    plt.xticks(digits)

    plt.ylim(0, 100)

    plt.show()


# Start application
window.mainloop()
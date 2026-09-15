def predict_digit():
    # Capture the drawing from canvas
    image = ImageGrab.grab(bbox=(
        canvas.winfo_rootx(),
        canvas.winfo_rooty(),
        canvas.winfo_rootx() + 280,
        canvas.winfo_rooty() + 280
    ))

    # Convert to grayscale
    image = image.convert("L")

    # Convert to NumPy array
    image_array = np.array(image)

    # Find the area containing the handwritten digit
    coords = np.argwhere(image_array > 30)

    if coords.size == 0:
        result_label.config(text="Please draw a digit first")
        return

    # Get bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop the digit
    cropped = image_array[y0:y1, x0:x1]

    # Make the image square
    h, w = cropped.shape
    size = max(h, w) + 20

    square = np.zeros((size, size), dtype=np.uint8)

    # Center the digit
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped

    # Resize to MNIST size
    image = Image.fromarray(square)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize
    image_array = np.array(image).astype("float32") / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    digit = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    # Display result
    result_label.config(
        text=f"Predicted Digit: {digit}\n"
             f"Confidence: {confidence:.2f}%"
    )

    # Show probability chart
     show_probability_chart(prediction)
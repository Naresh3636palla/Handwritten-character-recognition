import tkinter as tk
from tkinter import filedialog

window = tk.Tk()
window.title("Upload Test")
window.geometry("400x200")


def choose_file():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
            ("All files", "*.*")
        ]
    )

    if file_path:
        result.config(text="Selected:\n" + file_path)
    else:
        result.config(text="No file selected")


button = tk.Button(
    window,
    text="Upload Image",
    command=choose_file,
    font=("Arial", 14)
)

button.pack(pady=30)

result = tk.Label(
    window,
    text="No file selected",
    wraplength=350
)

result.pack()

window.mainloop()
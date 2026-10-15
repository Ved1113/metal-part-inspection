import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
import os

# Select image
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png"),
        ("All Files", "*.*")
    ]
)

if not image_path:
    print("No image selected.")
    raise SystemExit

# Load ONNX model
model = YOLO("exp6.onnx")

# Run inference
results = model(image_path, imgsz=640)

# Draw prediction
result_image = results[0].plot()

# Create output folder
os.makedirs("output", exist_ok=True)

# Save result
output_path = "output/onnx_result.jpg"
cv2.imwrite(output_path, result_image)

print("ONNX inference completed.")
print("Saved as:", output_path)
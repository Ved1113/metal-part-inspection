import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO


model = YOLO("exp6.onnx")


root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Metal Part Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png"),
        ("All Files", "*.*")
    ]
)

if not image_path:
    print("No image selected.")
    raise SystemExit


results = model(image_path, imgsz=640, verbose=False)


result = results[0]

if result.boxes is not None and len(result.boxes) > 0:
    confidence = float(result.boxes.conf[0])
    class_id = int(result.boxes.cls[0])
    defect_name = result.names[class_id]

    status = "FAIL"
else:
    confidence = 0.0
    defect_name = "No defect detected"
    status = "PASS"


result_image = result.plot()


os.makedirs("output", exist_ok=True)


output_path = "output/inspection_result.jpg"
cv2.imwrite(output_path, result_image)


print()
print("===== METAL PART INSPECTION =====")
print("Image      :", image_path)
print("Defect     :", defect_name)
print(f"Confidence : {confidence * 100:.2f}%")
print("Status     :", status)
print("Result     :", output_path)

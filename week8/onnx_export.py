from ultralytics import YOLO

model = YOLO("exp6.pt")

model.export(
    format="onnx",
    imgsz=640,
    simplify=True
)

print("ONNX export completed.")
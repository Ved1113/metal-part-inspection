# from ultralytics import YOLO

# image = "images/test.png"

# # YOLOv8
# model8 = YOLO("yolov8n.pt")
# results8 = model8(image)
# results8[0].show()

# # YOLO11
# model11 = YOLO("yolo11n.pt")
# results11 = model11(image)
# results11[0].show()
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

results = model("images/test.png")

results[0].show()
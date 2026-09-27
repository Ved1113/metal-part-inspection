from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")


results = model("images/parts_scene.png")


result_image = results[0].plot()


cv2.imwrite("output/yolo_result.jpg", result_image)

print("YOLO inference completed.")
print("Saved as output/yolo_result.jpg")
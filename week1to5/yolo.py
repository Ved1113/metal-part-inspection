from ultralytics import YOLO
import cv2

# Load a PRETRAINED model 
# This is NOT training - yolov8n.pt is already trained by Ultralytics
# on the COCO dataset (80 object categories: person, car, bus, dog, etc.)

model = YOLO('yolov8n.pt')

# Run inference on your image 
results = model('images/group.webp')

# Print what was detected 
for r in results:
    print('Objects detected:', len(r.boxes))
    for box in r.boxes:
        cls_id = int(box.cls[0])                  # numeric category ID
        class_name = model.names[cls_id]           # convert ID to readable name
        confidence = float(box.conf[0])             # how confident the model is
        print(f'  {class_name}: {confidence:.2f}')

#Draw and save the annotated result 
annotated = results[0].plot()   # auto-draws all boxes + labels
cv2.imwrite('yolo_result.jpg', annotated)
print('Saved as yolo_result.jpg')
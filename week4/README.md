# Week 4 — Features, Matching & Intro to Detection Models

## Objective

This week focuses on classical computer vision techniques used for feature matching, image alignment, object matching, and basic object detection. It also provides an introduction to pretrained YOLO models before moving into custom object detection in Month 2.

---

## Topics Covered

### 1. Template Matching

- Used OpenCV `matchTemplate()` to locate a template inside a larger image.
- Used normalized template matching to find the best matching location.
- Displayed the detected region using a bounding rectangle.
- Calculated and displayed the matching score.

**Implementation:** `parts.py` / `template_matching.py`

---

### 2. Homography and Image Alignment

- Used ORB feature detection to extract keypoints and descriptors.
- Used BFMatcher to match features between images.
- Used RANSAC to remove incorrect feature matches.
- Calculated the homography matrix.
- Used the homography for image alignment / panorama stitching.

**Implementation:** `panoramaa.py`

**Output:** `output/stitched_resultt.png`

---

### 3. Haar Cascade Face and Eye Detection

- Used OpenCV Haar Cascade classifiers.
- Detected faces from an input image.
- Detected eyes inside the detected face regions.
- Drawn bounding boxes around detected faces and eyes.

**Implementation:** `facedetection.py`

**Output:** `faces_detected.jpg`

---

### 4. Introduction to YOLO / Pretrained Object Detection

- Introduced the concept of pretrained object detection models.
- Used a pretrained YOLO model for inference.
- Loaded pretrained YOLO weights.
- Ran inference on an existing image.
- Generated an annotated image containing the model's detections.

**Implementation:** `yolo_inference.py`

**Input:** `images/parts_scene.png`

**Output:** `output/yolo_result.jpg`

> Note: YOLO is used only for pretrained inference in Week 4. Custom YOLO training is part of Week 6.

---

## Week 4 Task Completion

| Requirement | Implementation |
|---|---|
| Template matching | `parts.py` / `template_matching.py` |
| Homography / image alignment | `panoramaa.py` |
| Panorama / stitching demo | `panoramaa.py` |
| Haar face detection | `facedetection.py` |
| Haar eye detection | `facedetection.py` |
| Pretrained YOLO inference | `yolo_inference.py` |

---

## Folder Structure

```text
week4/
│
├── images/
│   ├── parts_scene.png
│   ├── nut_template.png
│   └── ...
│
├── output/
│   ├── stitched_resultt.png
│   ├── template_matching_result.jpg
│   └── yolo_result.jpg
│
├── parts.py
├── template_matching.py
├── panoramaa.py
├── facedetection.py
├── yolo_inference.py
├── faces_detected.jpg
└── README.md

## How to Run

Activate the virtual environment and run:

python facedetection.py

For panorama stitching:

python panoramaa.py

for template matching 
templatematch.py

for yolo inference 
yolo_inference.py

The generated results are saved inside the output folder.
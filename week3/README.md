# Week 3 — Feature Matching and Template Matching

## Objective

The objective of Week 3 was to understand traditional computer vision techniques for detecting and locating objects or machine parts without using deep learning models.

## Topics Covered

- Feature detection using ORB
- Feature description and matching
- Brute Force matching
- Lowe's ratio test
- Homography
- RANSAC
- Template matching
- Detection thresholding
- Non-Maximum Suppression (NMS)
- Drawing bounding boxes around detected objects

These techniques are useful for locating known objects or parts in images when a reference/template image is available.

---

## 1. Feature-Based Object Localization — `box.py`

Uses ORB feature detection to locate a reference object inside another scene image.

**Pipeline**

```
Template Image
   ↓
ORB Keypoint Detection
   ↓
Feature Descriptors
   ↓
Brute Force Matching
   ↓
Lowe's Ratio Test
   ↓
Homography
   ↓
RANSAC
   ↓
Perspective Transformation
   ↓
Object Boundary
```

**ORB (Oriented FAST and Rotated BRIEF)**
Detects distinctive points in an image and creates descriptors representing those points, which can be compared between two images.

**Brute Force Matcher**
Compares descriptors from the template image with descriptors from the scene image. `BFMatcher` with `NORM_HAMMING` is used because ORB produces binary descriptors.

**Lowe's Ratio Test**
For each feature, the two closest matches are considered. The best match is accepted only when it is sufficiently better than the second-best match:

```
m.distance < 0.75 × n.distance
```

**Homography**
Estimates the geometric transformation between the template and the corresponding region in the scene, allowing the program to locate the template even when its position or perspective changes.

**RANSAC (Random Sample Consensus)**
Removes incorrect matches (outliers) and keeps geometrically consistent matches. The template corners are then transformed and a boundary is drawn around the detected object.

**Result:**
```
Good matches: 42
RANSAC inliers: 40 out of 42
```

---

## 2. Template Matching — `parts.py`

Uses OpenCV template matching to locate multiple occurrences of a known part in a scene.

**Pipeline**

```
Scene Image + Template
   ↓
matchTemplate()
   ↓
Similarity Map
   ↓
Thresholding
   ↓
Candidate Bounding Boxes
   ↓
NMS
   ↓
Final Detections
```

**Template Matching**
`cv2.matchTemplate()` compares the template with different regions of the scene image, producing a similarity score at each location.

**Threshold**
A threshold of `0.65` was used to select candidate detections from the similarity map.

**Non-Maximum Suppression (NMS)**
Template matching can produce many overlapping detections around the same object. NMS removes overlapping duplicates and keeps the strongest detections.

**Result:**
```
Raw candidate detections: 27446
Final detections after NMS: 5
```

---

## 3. Key Concepts

| Concept | Description |
|---|---|
| **Feature Detection** | Identifies distinctive points in an image (corners, edges, texture patterns, other visually distinctive regions) that help recognize the same object in another image. |
| **Feature Descriptors** | Numerical representations of detected features, compared instead of comparing raw pixels. ORB generates binary descriptors compared using Hamming distance. |
| **Feature Matching** | Compares descriptors from two images to find corresponding features and determine whether the same object exists in the scene. |
| **Brute Force Matching** | Compares a descriptor from one image against all descriptors from another, keeping the closest matches by distance. |
| **Lowe's Ratio Test** | Rejects ambiguous matches by comparing the best match against the second-best match (`0.75` ratio used here). |
| **Homography** | A geometric transformation mapping one planar view of an object to another, giving its position/orientation in the scene. |
| **RANSAC** | Identifies matches that follow a consistent geometric transformation and rejects inconsistent outliers. |
| **matchTemplate()** | Slides a template across a scene image and computes a similarity score at each position. |
| **Threshold** | Keeps only candidate locations with sufficiently high similarity scores. |
| **NMS** | Removes overlapping duplicate bounding boxes, keeping the strongest detection per object. |

---

## 4. Input Images (`images/`)

| File | Description |
|---|---|
| `box.png` | Reference template used for feature matching |
| `box_scene.png` | Scene image containing the reference object |
| `nut_template.png` | Reference template of the machine part |
| `parts_scene.png` | Scene image containing multiple occurrences of the part |

## 5. Output (`output/`)

| File | Description |
|---|---|
| `box_result.png` | Localized template object using ORB, feature matching, homography, RANSAC and perspective transformation |
| `parts_result.png` | Final detected parts after template matching, thresholding, bounding-box generation and NMS |

---

## Overall Week 3 Pipeline

```
                 Traditional Computer Vision
                           │
             ┌─────────────┴─────────────┐
             │                           │
      Feature Matching            Template Matching
             │                           │
            ORB                  matchTemplate()
             │                           │
       Descriptors                 Similarity
             │                           │
       BFMatcher                  Threshold
             │                           │
     Lowe's Ratio Test          Candidate Boxes
             │                           │
       Homography                     NMS
             │                           │
          RANSAC                  Final Detections
             │
      Object Localization
```

## Folder Structure

```text
week3/
├── images/
│   ├── box.png
│   ├── box_scene.png
│   ├── nut_template.png
│   └── parts_scene.png
├── output/
│   ├── box_result.png
│   └── parts_result.png
├── box.py
├── parts.py
├── requirements.txt
├── README.md
└── venv/
```

## How to Run

**1. Activate the virtual environment** (Windows PowerShell)

```
venv\Scripts\activate
```

`(venv)` should appear in the terminal once activated.

**2. Install dependencies**

```
pip install -r requirements.txt
```

**3. Run feature matching**

```
python box.py
```

Result saved to `output/box_result.png`. Terminal output:
```
Good matches: 42
RANSAC inliers: 40 out of 42
```

**4. Run template matching**

```
python parts.py
```

Result saved to `output/parts_result.png`. Terminal output:
```
Raw candidate detections: 27446
Final detections after NMS: 5
```

---

## Feature Matching vs. Template Matching

**Feature Matching**
Looks for distinctive points and compares their descriptors. Useful when:
- The object can change position or rotate
- Perspective can change
- The object has distinctive features

Advantage: it does not require the entire object to have exactly the same pixel appearance.

**Template Matching**
Directly compares a reference image with regions of the scene. Useful when:
- The object's appearance is relatively consistent
- The scale is similar
- A reference template is available
- The object has a consistent visual pattern

## Traditional Computer Vision vs. Deep Learning

**Traditional Computer Vision**
```
Image → Hand-designed features/template → Matching → Detection
```

**Deep Learning**
```
Image → Neural Network → Learned Features → Object Detection
```

Traditional methods work well when object appearance and conditions are controlled. Deep-learning detectors are generally more suitable when there is variation in lighting, position, scale, rotation, background, or object appearance. The Week 3 experiments provide a foundation for understanding why modern object detection models (e.g. YOLO) are useful for industrial inspection.

## Learning Outcome

By completing Week 3, I understood two traditional computer vision approaches for object localization: feature matching, which uses distinctive image features and geometric relationships, and template matching, which directly compares a known template against regions of an image.

I also learned:
- How ORB detects and describes features
- How Brute Force matching compares descriptors
- How Lowe's ratio test removes ambiguous matches
- How homography maps a template into a scene
- How RANSAC removes incorrect feature matches
- How template matching generates similarity scores
- How thresholding selects candidate detections
- How NMS removes duplicate bounding boxes

These techniques provide a foundation for understanding more advanced object detection methods such as YOLO and other deep-learning-based detectors.

## Technologies Used

- Python
- OpenCV
- NumPy
- VS Code
- Python Virtual Environment

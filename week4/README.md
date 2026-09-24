# Week 4 — Advanced OpenCV

This week focused on advanced OpenCV techniques used for face detection,
feature matching, and image stitching.

## Topics Covered

- Haar Cascade face detection
- Haar Cascade eye detection
- ORB feature detection
- ORB feature descriptors
- Brute-Force feature matching
- Homography
- RANSAC
- Panorama stitching

---

## 1. Face and Eye Detection

### File
`facedetection.py`

### Input
`images/group.webp`

### Method

Haar Cascade classifiers were used for detecting faces and eyes.

Two pretrained Haar Cascade files were used:

- `haarcascade_frontalface_alt2.xml`
- `haarcascade_eye.xml`

The input image is first converted to grayscale because Haar Cascade
detection works efficiently on grayscale images.

The face detector identifies faces and draws a green bounding box around
each detected face.

For every detected face, the corresponding face region is extracted and
the eye detector is applied to that region.

Detected eyes are marked with blue bounding boxes.

### Output

`output/faces_detected.jpg`

---

## 2. ORB Feature Matching

### File
`panoramaa.py`

### Input

- `images/photoA.png`
- `images/photoB.png`

ORB (Oriented FAST and Rotated BRIEF) is used to detect important
features or keypoints in both images.

ORB generates descriptors for these keypoints so that corresponding
features between the two images can be matched.

A Brute-Force Matcher with Hamming distance is used because ORB produces
binary descriptors.

The best matches are selected based on descriptor distance.

---

## 3. Homography and RANSAC

After finding matching points between the two images, homography is used
to calculate the geometric transformation between them.

RANSAC is used during homography estimation to reduce the effect of
incorrect feature matches (outliers).

The detected transformation is then used to warp one image into the
coordinate system of the other image.

---

## 4. Panorama Stitching

After calculating the transformation, the second image is warped and
combined with the first image to create a panorama.

### Outputs

- `output/orb_matches.png`
- `output/stitched_resultt.png`

`orb_matches.png` shows the feature correspondences between the two
images.

`stitched_resultt.png` contains the final stitched panorama.

---

## Folder Structure

```text
week_4/
│
├── images/
│   ├── group.webp
│   ├── photoA.png
│   └── photoB.png
│
├── output/
│   ├── faces_detected.jpg
│   ├── orb_matches.png
│   └── stitched_resultt.png
│
├── facedetection.py
├── panoramaa.py
├── haarcascade_eye.xml
├── haarcascade_frontalface_alt2.xml
├── requirements.txt
├── README.md
└── venv/

## Requirements
Python
OpenCV
NumPy

The project was developed and tested using a Python virtual environment.

## How to Run

Activate the virtual environment and run:

python facedetection.py

For panorama stitching:

python panoramaa.py

The generated results are saved inside the output folder.
### Week 3 — Contours, Shape & Object Analysis

# Objective

The objective of Week 3 was to understand contour-based object analysis using OpenCV.

The main task was to take an image containing scattered objects and:

Detect individual objects

Count the objects

Analyze their size and shape

Filter objects using area

Calculate aspect ratio

Draw bounding boxes

Draw minimum enclosing circles

Classify objects based on size and shape


# Topics Covered

Contour detection

Contour hierarchy

Bounding rectangles

Minimum enclosing circles

Contour area

Aspect ratio

Circularity

Object counting

Area-based filtering

Size classification

Shape classification


# How to Run

1. Activate the virtual environment

Windows PowerShell:

venv\Scripts\activate

After activation:

(venv)

should appear in the terminal.

2. Install dependencies

pip install -r requirements.txt

3. Run the object analysis

python object_analysis.py

The program performs:

Image
 ↓
HSV conversion
 ↓
Background segmentation
 ↓
Binary mask
 ↓
Morphological processing
 ↓
Contour detection
 ↓
Area filtering
 ↓
Bounding box
 ↓
Minimum enclosing circle
 ↓
Area / aspect ratio / circularity analysis
 ↓
Size classification
 ↓
Shape classification
 ↓
Final annotated image


Folder Structure

week3/

├── images/
│   └── scattered_objects.jpg
│
├── output/
│   ├── object_analysis.jpg
│   └── object_mask.jpg
│
├── object_analysis.py
├── requirements.txt
├── README.md
└── venv/

# Technologies Used

Python

OpenCV

NumPy

VS Code

Python Virtual Environment
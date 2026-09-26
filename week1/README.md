# Week 1 - Image and Video Basics

## Overview

This week covers the OpenCV and NumPy foundations required for image and video processing.

The work follows the official Week 1 task: image I/O, image arrays, color spaces, basic transformations, drawing, image arithmetic, and webcam/video handling. It also includes the required batch-processing mini task.

## Topics Covered

- Image reading and writing: `imread`, `imshow`, `imwrite`
- Image as a NumPy array: shape, dimensions, channels and pixel values
- Color spaces: BGR, RGB, HSV and grayscale
- Resizing
- Cropping and ROI slicing
- Rotation
- Horizontal and vertical flipping
- Drawing lines, rectangles, circles and text
- Bitwise operations and masking
- Image blending using `addWeighted`
- Webcam capture
- Video recording
- Batch image processing

## Files

| File | Description |
|---|---|
| `main.py` | Reads an image and prints its shape and dimensions |
| `week1.py` | Examines image dimensions, channels, pixels, BGR values and grayscale values |
| `grayscale.py` | Converts one image to grayscale and saves it |
| `resize.py` | Resizes one image and saves it |
| `cropped.py` | Crops a region of interest using NumPy slicing |
| `rotate.py` | Rotates an image 90 degrees clockwise |
| `flip.py` | Creates horizontal and vertical flips |
| `color_spaces.py` | Demonstrates BGR to RGB, HSV and grayscale conversion |
| `linedraw.py` | Draws a line on an image |
| `rectangle.py` | Draws a rectangle |
| `circle.py` | Draws a circle |
| `text.py` | Adds text to an image |
| `bitwise.py` | Demonstrates masking and bitwise AND |
| `blending.py` | Demonstrates image blending with `addWeighted` |
| `batch_process.py` | Reads all supported images from a folder, converts them to grayscale, resizes them and saves the batch output |
| `capture.py` | Captures frames from a webcam |
| `videosave.py` | Records webcam video to an AVI file |

## Batch Processing Mini Task

The required mini task is implemented in `batch_process.py`.

Input folder:

```text
images/batch_input/
```

For every supported image, the script:

1. Reads the image.
2. Converts it from BGR to grayscale.
3. Resizes it to `500 x 500` pixels.
4. Saves the processed image in:

```text
output/batch/
```

This demonstrates batch processing instead of processing only one image manually.

## How to Run

Open the Week 1 folder in VS Code and activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies if required:

```powershell
pip install -r requirements.txt
```

Run the required batch task:

```powershell
python batch_process.py
```

The terminal prints each processed file and the final number of successfully saved images.

Other scripts can be run individually, for example:

```powershell
python color_spaces.py
python blending.py
python linedraw.py
```

Scripts that use `cv2.imshow()` require a desktop environment and wait for a key press before closing.

## Folder Structure

```text
week_1new/
├── images/
│   ├── sample.jpg
│   └── batch_input/
├── output/
│   └── batch/
├── batch_process.py
├── blending.py
├── bitwise.py
├── capture.py
├── circle.py
├── color_spaces.py
├── cropped.py
├── flip.py
├── grayscale.py
├── linedraw.py
├── main.py
├── rectangle.py
├── resize.py
├── rotate.py
├── text.py
├── videosave.py
├── week1.py
├── requirements.txt
├── README.md
└── venv/
```

# 🔍 Metal Part Inspection

Automatic defect detection for metal parts using a YOLO model.
Detects **Crack**, **Porosity** and **Scratch** from images, videos and a live camera,
and gives a **PASS / FAIL** result.

## Features

- **Single image** and **batch** inspection with defect counts per type
- **Video inspection**: every frame is checked, each defect is counted once,
  and the clearest frame of every defect is saved as evidence (with frame number and time)
- **Live camera** inspection with live and session counts
- **Login and Register** with role-based access (admin / user)
- **SQLite database** for users, inspection history and defect records
- **Admin dashboard**: totals, defect rate, charts, user management, CSV export
- Passwords are stored as **salted PBKDF2 hashes**

## Tech stack

Python · Streamlit · Ultralytics YOLO (ONNX) · OpenCV · SQLite · Pandas · streamlit-webrtc
   

## Project structure

```
week8/
├── streamlit_metal_inspection.py       # main Streamlit web application
├── onnx_export.py         # exports exp6.pt to exp6.onnx
├── onnx_predict.py        # desktop script: pick one image and predict with the ONNX model
├── exp6.pt                # trained YOLO weights (PyTorch)
├── exp6.onnx              # exported ONNX model used by the app
├── test.jpg               # sample test image
├── requirements.txt
├── README.md
├── inspection.db          # created automatically (not in Git)
├── output/                # processed videos and defect images
│
│   # Personal test files 
├── metal_inspection.py                 # early desktop prototype
└── benchmark.py           # model speed test
```

project/
├── metal_inspection.py                 # Streamlit app (main user interface)
├── pipeline.py            # Shared logic: preprocessing, detection, severity, alert, report
├── api.py                 # FastAPI service + mini demo page
├── production_line.py     # Headless camera -> decision -> log -> alert script
├── export_optimize.py     # ONNX export, INT8 quantization, benchmark, mAP comparison
├── exp6.onnx              # FP32 model
├── exp6_int8.onnx         # INT8 quantized model (created by export_optimize.py)
├── calib_images/          # 30-50 sample part images used for quantization
├── inspection.db          # SQLite database (created automatically)
└── output/                # Results, alerts.jsonl, line_log.csv, defect images, processed videos


│   # Personal test files 
├── metal_inspection.py                 # early desktop prototype
└── benchmark.py           # model speed test
├── appp.py                
└── streamlit_metal_inspection.py          
```
```
## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Place your trained model file `exp6.onnx` in the project folder.
(If the file is larger than 100 MB, GitHub will not accept it. Use Git LFS or a GitHub Release.)

Installation

Python 3.10 or newer is recommended.

bash
pip install streamlit streamlit-webrtc ultralytics opencv-python numpy pandas pillow av
pip install onnxruntime onnx onnxsim
pip install fastapi uvicorn python-multipart

Optional: ffmpeg (or pip install imageio-ffmpeg) so processed videos play in the browser.

Put your trained model exp6.onnx in the project folder.

Usage
1. Streamlit app (main interface)
bash
streamlit run metal_inspection.py

Open http://localhost:8501. For webcam use, use localhost and allow camera permission.

Default admin login: admin / admin123. Change this password in the Account page.

Pages:

Home: your stats and recent inspections
Single Image: part ID, CLAHE option, PASS/FAIL, severity, alert, report download
Batch Inspection: many images at once, CSV report
Uploaded Video: per-defect evidence frames, defect log, processed video
Live Camera: real-time webcam detection
Pipeline Info: model speed benchmark, line-speed calculator, pipeline and deployment notes
History: your past inspections
Dashboard / Manage Users: admin only

2. FastAPI service
bash
uvicorn api:app --port 8000
http://localhost:8000 : simple upload page
http://localhost:8000/docs : automatic API test page
POST /inspect : send an image, get the verdict as JSON
GET /alerts : recent alerts
GET /health : health check

Example:

bash
curl -F file=@part.jpg -F part_id=P001 http://localhost:8000/inspect

Response includes part_id, verdict, severity, counts, detections, processing_ms, alert and the annotated image as base64.

## Run

```bash
streamlit run streamlit_metal_inspection.py
```

Open http://localhost:8501. For the webcam, use `localhost` and allow camera permission.

## Default login

| Username | Password | Role |
|---|---|---|
| admin | admin123 | admin |

This is a demo account. **Change the password** in *Account → Change Password* after the first login.
New users can sign up from the **Register** tab and get the *user* role. Only an admin can promote users.

## How video defect counting works

1. Every frame is checked by the YOLO model.
2. A tracker matches boxes between frames (box overlap), so one defect seen in many frames is counted **once**.
3. When the picture changes to a new part or image (scene change), the tracker resets, so the next defect counts as a **new** one.
4. Two settings on the video page tune this:
   - **Frames a defect must appear in**: higher = fewer false alarms.
   - **Scene-change sensitivity**: lower = a new part is detected more easily.

## Database

| Table | Stores |
|---|---|
| `users` | username, password hash, salt, role |
| `inspections` | one row per inspection: user, file, PASS/FAIL, counts, time |
| `defects` | one row per video defect: type, frame, time, confidence, image path |

Defect images are saved as files in `output/defects/`, and the database stores their paths.


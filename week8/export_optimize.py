"""
Week 8 - model optimization: ONNX export, INT8 quantization, speed + accuracy comparison.

Usage:
  python export_optimize.py quant                 # make exp6_int8.onnx + speed table
  python export_optimize.py export quant          # (needs exp6.pt) re-export then quantize
  python export_optimize.py map data.yaml         # compare mAP / Crack recall of FP32 vs INT8
  python export_optimize.py                       # just benchmark what exists
"""
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType)

PT = "exp6.pt"
FP32 = "exp6.onnx"
INT8 = "exp6_int8.onnx"
CALIB_DIR = "calib_images"      # 30-50 representative part images
IMGSZ = 640


def export_fp32():
    from ultralytics import YOLO
    YOLO(PT).export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)


def load_input(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None]


class Reader(CalibrationDataReader):
    def __init__(self, folder, input_name):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.files = ([os.path.join(folder, f) for f in sorted(os.listdir(folder))
                       if f.lower().endswith(exts)][:100] if os.path.isdir(folder) else [])
        self.name, self.i = input_name, 0

    def get_next(self):
        while self.i < len(self.files):
            path = self.files[self.i]
            self.i += 1
            if cv2.imread(path) is not None:
                return {self.name: load_input(path)}
        return None


def quantize_int8():
    sess = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    reader = Reader(CALIB_DIR, name)
    if not reader.files:
        sys.exit(f"ERROR: put 30-50 images (.jpg/.png) in the '{CALIB_DIR}' folder "
                 f"next to this script, then run again.")
    print(f"Calibrating with {len(reader.files)} images...")
    quantize_static(FP32, INT8, reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8)
    print("Saved", INT8)


def benchmark(path, runs=50):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    for _ in range(5):
        sess.run(None, {name: x})
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - s) * 1000)
    ms = float(np.mean(t))
    print(f"{path:18s} {os.path.getsize(path) / 1e6:6.1f} MB | {ms:7.1f} ms | {1000 / ms:5.1f} FPS")
    return ms


def compare_map(data_yaml):
    from ultralytics import YOLO
    for p in (FP32, INT8):
        if not os.path.exists(p):
            continue
        m = YOLO(p, task="detect").val(data=data_yaml, imgsz=IMGSZ, verbose=False)
        print(f"\n{p}: mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  "
              f"precision={m.box.mp:.3f}  recall={m.box.mr:.3f}")
        for i, r in enumerate(m.box.r):          # per-class recall (Crack = class 0)
            print(f"   class {i} recall: {r:.3f}")


def takt_budget(parts_per_min):
    print(f"Line speed {parts_per_min} parts/min -> {60000 / parts_per_min:.0f} ms available per part")


if __name__ == "__main__":
    if "export" in sys.argv:
        export_fp32()
    if "quant" in sys.argv:
        quantize_int8()
    if "map" in sys.argv:
        i = sys.argv.index("map")
        compare_map(sys.argv[i + 1] if len(sys.argv) > i + 1 else "data.yaml")
    else:
        print("\n=== SPEED / SIZE ===")
        ms32 = benchmark(FP32)
        if os.path.exists(INT8):
            ms8 = benchmark(INT8)
            print(f"INT8 speed-up: {ms32 / ms8:.2f}x")
        takt_budget(30)
        takt_budget(60)
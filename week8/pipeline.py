import os, io, json, time, base64, uuid, threading, urllib.request
from html import escape
import cv2, numpy as np
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "exp6.onnx")
OUTPUT_DIR = "output"
ALERT_FILE = os.path.join(OUTPUT_DIR, "alerts.jsonl")
ALERT_WEBHOOK = os.environ.get("ALERT_WEBHOOK", "")   # optional Slack/Teams/etc URL
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Crack", 1: "Porosity", 2: "Scratch"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 165, 255), 2: (255, 0, 0)}   # BGR
CLASS_WEIGHT = {0: 3.0, 1: 2.0, 2: 1.0}      # Crack is the most critical
SEV_ORDER = ["None", "Minor", "Major", "Critical"]
SEV_COLORS = {"None": "#15803d", "Minor": "#ca8a04", "Major": "#ea580c", "Critical": "#b91c1c"}

MAX_SIDE = 960          # big photos are shrunk to this before anything else (speed)

_model = None
_lock = threading.RLock()     # ONE model shared by every page; one prediction at a time


# ---------- model (loaded once, warmed up once) ----------
def get_model():
    global _model
    with _lock:
        if _model is None:
            _model = YOLO(MODEL_PATH, task="detect")
        return _model


def warm_up():
    """Run one dummy prediction so the FIRST real image is not slow."""
    t = time.perf_counter()
    predict(np.zeros((640, 640, 3), np.uint8), conf=0.25, imgsz=640)
    return round((time.perf_counter() - t) * 1000, 1)


def predict(img, conf=0.25, imgsz=640, iou=0.7, max_det=300):
    """Thread-safe prediction used by the app, API and line script."""
    with _lock:
        return get_model().predict(source=img, conf=conf, iou=iou, max_det=max_det,
                                   imgsz=imgsz, verbose=False)[0]


def new_part_id():
    return "P" + time.strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:3].upper()


# ---------- 1. preprocessing ----------
def preprocess(bgr, max_side=MAX_SIDE, enhance=False):
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if enhance:                                  # CLAHE helps with uneven lighting on metal
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return bgr


# ---------- 2. severity (tunable thresholds) ----------
def severity(cls, conf, box, shape):
    H, W = shape
    area_ratio = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / float(H * W)
    score = CLASS_WEIGHT.get(cls, 1.0) * conf * (1 + 10 * area_ratio)
    return "Critical" if score >= 2.5 else "Major" if score >= 1.2 else "Minor"


# ---------- 3. detect + decide ----------
def inspect_image(bgr, part_id=None, conf=0.25, imgsz=640, enhance=False):
    t0 = time.perf_counter()
    part_id = part_id or new_part_id()
    img = preprocess(bgr, enhance=enhance)
    t1 = time.perf_counter()
    r = predict(img, conf=conf, imgsz=imgsz)
    t2 = time.perf_counter()
    H, W = img.shape[:2]

    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                           r.boxes.conf.cpu().numpy()):
            c, s, box = int(c), float(s), [float(v) for v in b]
            dets.append({"cls": c, "type": CLASS_NAMES.get(c, str(c)),
                         "confidence": round(s, 3),
                         "severity": severity(c, s, box, (H, W)),
                         "box": [round(v, 1) for v in box]})

    counts = {k: 0 for k in CLASS_NAMES}
    for d in dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    total = len(dets)
    sev = max((d["severity"] for d in dets), key=SEV_ORDER.index, default="None")

    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        col = CLASS_COLORS.get(d["cls"], (0, 255, 0))
        label = f'{d["type"]} {d["confidence"]:.2f} [{d["severity"]}]'
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    verdict = "FAIL" if total else "PASS"
    cv2.rectangle(out, (0, 0), (230, 34), (0, 0, 200) if total else (0, 150, 0), -1)
    cv2.putText(out, f"{part_id}: {verdict}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    t3 = time.perf_counter()

    return {"part_id": part_id, "verdict": verdict, "severity": sev, "counts": counts,
            "total": total, "types": sum(1 for v in counts.values() if v > 0),
            "detections": dets, "annotated": out,
            "avg_conf": float(np.mean([d["confidence"] for d in dets])) if dets else 0.0,
            "processing_ms": round((t3 - t0) * 1000, 1),
            "timings": {"preprocess_ms": round((t1 - t0) * 1000, 1),
                        "model_ms": round((t2 - t1) * 1000, 1),
                        "draw_ms": round((t3 - t2) * 1000, 1)},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------- 4. alert ----------
def raise_alert(res, min_level="Major"):
    """Log (and optionally POST) an alert for FAIL parts at/above min_level."""
    if res["verdict"] != "FAIL" or SEV_ORDER.index(res["severity"]) < SEV_ORDER.index(min_level):
        return False
    rec = {"time": res["timestamp"], "part_id": res["part_id"], "severity": res["severity"],
           "defects": res["counts"] and {CLASS_NAMES[k]: v for k, v in res["counts"].items() if v},
           "total": res["total"]}
    with open(ALERT_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"🚨 ALERT {rec}")
    if ALERT_WEBHOOK:
        try:
            req = urllib.request.Request(ALERT_WEBHOOK, json.dumps({"text": f"🚨 {rec}"}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass
    return True


def load_alerts(n=20):
    if not os.path.exists(ALERT_FILE):
        return []
    with open(ALERT_FILE) as f:
        return [json.loads(l) for l in f.readlines()[-n:]][::-1]


# ---------- 5. summary report (single self-contained HTML; print to PDF from browser) ----------
def jpeg_b64(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def build_report_html(res, operator="-", file_name="-"):
    col = "#15803d" if res["verdict"] == "PASS" else "#b91c1c"
    rows = "".join(
        f"<tr><td>{i}</td><td>{escape(d['type'])}</td><td>{d['confidence']*100:.0f}%</td>"
        f"<td style='color:{SEV_COLORS[d['severity']]};font-weight:700'>{d['severity']}</td></tr>"
        for i, d in enumerate(res["detections"], 1)) or "<tr><td colspan=4>No defects</td></tr>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Inspection {escape(res['part_id'])}</title>
<style>body{{font-family:Arial;max-width:820px;margin:30px auto;color:#0f172a}}
.v{{background:{col};color:#fff;padding:18px 24px;border-radius:12px;font-size:28px;font-weight:800}}
table{{border-collapse:collapse;width:100%;margin-top:14px}}td,th{{border:1px solid #cbd5e1;padding:8px;text-align:left}}
th{{background:#f1f5f9}}img{{width:100%;border-radius:10px;margin-top:16px}}</style></head><body>
<h2>Metal Part Inspection Report</h2>
<div class="v">{res['verdict']} &nbsp; <span style="font-size:16px">Severity: {res['severity']}</span></div>
<table><tr><th>Part ID</th><td>{escape(res['part_id'])}</td><th>Time</th><td>{res['timestamp']}</td></tr>
<tr><th>File</th><td>{escape(file_name)}</td><th>Operator</th><td>{escape(operator)}</td></tr>
<tr><th>Crack / Porosity / Scratch</th><td>{res['counts'][0]} / {res['counts'][1]} / {res['counts'][2]}</td>
<th>Processing time</th><td>{res['processing_ms']} ms</td></tr></table>
<h3>Defect list</h3><table><tr><th>#</th><th>Type</th><th>Confidence</th><th>Severity</th></tr>{rows}</table>
<img src="data:image/jpeg;base64,{jpeg_b64(res['annotated'])}"></body></html>"""


# ---------- 6. video inspection (used by the FastAPI /inspect_video route) ----------
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inspect_video(path, conf=0.25, stride=2, min_hits=2, scene_threshold=25.0,
                  max_age=8, iou_thr=0.3, max_seconds=None):
    """Check a video. Every distinct defect is counted ONCE (tracked across frames) and its
    clearest frame is kept as evidence. Returns verdict, severity, counts and defect list."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    stride = max(1, int(stride))
    t0 = time.perf_counter()

    tracks, registry = {}, {}
    next_id = 1
    counts = {k: 0 for k in CLASS_NAMES}
    prev_small = None
    fi = checked = det_frames = scene_changes = 0
    H = W = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if max_seconds and fi / fps > max_seconds:
            break
        if fi % stride:
            continue
        checked += 1
        H, W = frame.shape[:2]

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        if prev_small is not None and float(np.mean(np.abs(small - prev_small))) > scene_threshold:
            for t in tracks.values():
                if t["hits"] < min_hits:
                    t["jpg"] = None
            tracks = {}
            scene_changes += 1
        prev_small = small

        r = predict(frame, conf=conf, imgsz=640)
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                               r.boxes.conf.cpu().numpy()):
                dets.append((int(c), float(s), [float(v) for v in b]))
        if dets:
            det_frames += 1

        # match detections to existing tracks (same class, highest IoU first)
        pairs = []
        for di, (c, _, b) in enumerate(dets):
            for tid, t in tracks.items():
                if t["cls"] == c:
                    v = _iou(b, t["box"])
                    if v >= iou_thr:
                        pairs.append((v, di, tid))
        pairs.sort(reverse=True)
        assigned, used = [None] * len(dets), set()
        for _, di, tid in pairs:
            if assigned[di] is None and tid not in used:
                assigned[di] = tid
                used.add(tid)

        for di, (c, s, b) in enumerate(dets):
            tid = assigned[di]
            if tid is None:
                tid = next_id
                next_id += 1
                tracks[tid] = registry[tid] = {
                    "cls": c, "box": b, "hits": 0, "last": checked, "no": None,
                    "best_conf": 0.0, "jpg": None, "best_box": None, "best_frame": None}
            t = tracks[tid]
            t["box"], t["last"] = b, checked
            t["hits"] += 1
            if t["jpg"] is None or s > t["best_conf"] + 0.02:
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok2:
                    t["jpg"], t["best_box"], t["best_frame"] = buf.tobytes(), b, fi
            t["best_conf"] = max(t["best_conf"], s)
            if t["hits"] == min_hits:
                counts[c] = counts.get(c, 0) + 1
                t["no"] = sum(counts.values())

        for tid in [k for k, t in tracks.items() if checked - t["last"] > max_age]:
            if tracks[tid]["hits"] < min_hits:
                tracks[tid]["jpg"] = None          # free memory for false alarms
            del tracks[tid]

    cap.release()

    defects = []
    for t in sorted((t for t in registry.values() if t["no"] is not None), key=lambda t: t["no"]):
        sev_t = severity(t["cls"], t["best_conf"], t["best_box"], (H, W))
        img = cv2.imdecode(np.frombuffer(t["jpg"], np.uint8), cv2.IMREAD_COLOR) if t["jpg"] else None
        if img is not None:
            x1, y1, x2, y2 = [int(v) for v in t["best_box"]]
            col = CLASS_COLORS.get(t["cls"], (0, 255, 0))
            label = f'#{t["no"]} {CLASS_NAMES.get(t["cls"], t["cls"])} {t["best_conf"]:.2f} [{sev_t}]'
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            cv2.putText(img, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, col, 2, cv2.LINE_AA)
        defects.append({"no": t["no"], "type": CLASS_NAMES.get(t["cls"], str(t["cls"])),
                        "cls": t["cls"], "confidence": round(t["best_conf"], 3),
                        "severity": sev_t, "frame": int(t["best_frame"] or 0),
                        "time_sec": round((t["best_frame"] or 0) / fps, 2),
                        "box": [round(v, 1) for v in t["best_box"]], "annotated": img})

    total = sum(counts.values())
    sev = max((d["severity"] for d in defects), key=SEV_ORDER.index, default="None")
    return {"verdict": "FAIL" if total else "PASS", "severity": sev, "counts": counts,
            "total": total, "defects": defects, "frames_total": fi, "frames_checked": checked,
            "detection_frames": det_frames, "scene_changes": scene_changes,
            "fps": round(fps, 2), "processing_s": round(time.perf_counter() - t0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
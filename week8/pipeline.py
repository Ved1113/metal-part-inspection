import os, io, json, time, base64, uuid, urllib.request
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

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH, task="detect")
    return _model


def new_part_id():
    return "P" + time.strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:3].upper()


# ---------- 1. preprocessing ----------
def preprocess(bgr, max_side=1280, enhance=False):
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
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
    r = get_model().predict(source=img, conf=conf, imgsz=imgsz, verbose=False)[0]
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

    return {"part_id": part_id, "verdict": verdict, "severity": sev, "counts": counts,
            "total": total, "types": sum(1 for v in counts.values() if v > 0),
            "detections": dets, "annotated": out,
            "avg_conf": float(np.mean([d["confidence"] for d in dets])) if dets else 0.0,
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
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
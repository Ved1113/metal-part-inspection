import io
import os
import re
import time
import hmac
import shutil
import sqlite3
import zipfile
import hashlib
import tempfile
import threading
import subprocess
from contextlib import closing
from html import escape

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "exp6.onnx"
OUTPUT_DIR = "output"
DEFECT_DIR = os.path.join(OUTPUT_DIR, "defects")
DB_FILE = "inspection.db"
OLD_CSV = "inspection_history.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEFECT_DIR, exist_ok=True)

CLASS_NAMES = {0: "Crack", 1: "Porosity", 2: "Scratch"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 165, 255), 2: (255, 0, 0)}  # BGR

INSPECTION_COLUMNS = [
    "username", "inspection_type", "file_name", "result",
    "crack", "porosity", "scratch", "total_defects",
    "defect_types", "processing_time", "timestamp",
]

st.set_page_config(page_title="Metal Part Inspection", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH, task="detect")


model = load_model()


# ============================================================
# DATABASE (SQLite)
# ============================================================

def get_conn():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def db_exec(query, params=()):
    with closing(get_conn()) as conn:
        with conn:
            return conn.execute(query, params).lastrowid


def db_query(query, params=()):
    with closing(get_conn()) as conn:
        return pd.read_sql_query(query, conn, params=params)


def hash_password(password, salt=None):
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt), 200_000).hex()
    return digest, salt


def create_user(username, password, role="user"):
    pw_hash, salt = hash_password(password)
    try:
        db_exec(
            "INSERT INTO users (username, password_hash, salt, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, pw_hash, salt, role, time.strftime("%Y-%m-%d %H:%M:%S")))
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate(username, password):
    """Return role if credentials are correct, else None."""
    df = db_query("SELECT password_hash, salt, role FROM users WHERE username = ?", (username,))
    if df.empty:
        return None
    row = df.iloc[0]
    digest, _ = hash_password(password, row["salt"])
    if hmac.compare_digest(digest, row["password_hash"]):
        return row["role"]
    return None


def init_db():
    with closing(get_conn()) as conn:
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                inspection_type TEXT,
                file_name TEXT,
                result TEXT,
                crack INTEGER, porosity INTEGER, scratch INTEGER,
                total_defects INTEGER, defect_types INTEGER,
                processing_time REAL,
                timestamp TEXT,
                part_id TEXT,
                severity TEXT
            );
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL
                    REFERENCES inspections(id) ON DELETE CASCADE,
                defect_no INTEGER,
                defect_type TEXT,
                frame_no INTEGER,
                time_sec REAL,
                confidence REAL,
                image_path TEXT,
                severity TEXT
            );
            CREATE TABLE IF NOT EXISTS alert_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER,
                part_id TEXT,
                alert_type TEXT NOT NULL,
                severity TEXT,
                message TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE SET NULL
            );
            """)
            # Backward-compatible migration for existing SQLite databases.
            inspection_cols = {r[1] for r in conn.execute("PRAGMA table_info(inspections)")}
            if "part_id" not in inspection_cols:
                conn.execute("ALTER TABLE inspections ADD COLUMN part_id TEXT")
            if "severity" not in inspection_cols:
                conn.execute("ALTER TABLE inspections ADD COLUMN severity TEXT")
            defect_cols = {r[1] for r in conn.execute("PRAGMA table_info(defects)")}
            if "severity" not in defect_cols:
                conn.execute("ALTER TABLE defects ADD COLUMN severity TEXT")

    if db_query("SELECT COUNT(*) AS n FROM users")["n"][0] == 0:
        create_user("admin", "admin123", "admin")      # change after first login!

    # one-time import of the old CSV history
    if os.path.exists(OLD_CSV) and db_query("SELECT COUNT(*) AS n FROM inspections")["n"][0] == 0:
        try:
            old = pd.read_csv(OLD_CSV)
            if len(old) and set(INSPECTION_COLUMNS).issubset(old.columns):
                with closing(get_conn()) as conn:
                    old[INSPECTION_COLUMNS].to_sql("inspections", conn,
                                                   if_exists="append", index=False)
            shutil.move(OLD_CSV, OLD_CSV + ".migrated")
        except Exception:
            pass


def save_history(username, inspection_type, file_name, result, counts, processing_time,
                 part_id="", severity="None"):
    crack, porosity, scratch = (int(counts.get(k, 0)) for k in (0, 1, 2))
    return db_exec(
        "INSERT INTO inspections (username, inspection_type, file_name, result, crack, "
        "porosity, scratch, total_defects, defect_types, processing_time, timestamp, part_id, severity) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (username, inspection_type, file_name, result, crack, porosity, scratch,
         crack + porosity + scratch,
         sum(1 for v in (crack, porosity, scratch) if v > 0),
         round(processing_time, 2), time.strftime("%Y-%m-%d %H:%M:%S"),
         part_id.strip(), severity))


def save_defects(inspection_id, defects):
    for d in defects:
        db_exec(
            "INSERT INTO defects (inspection_id, defect_no, defect_type, frame_no, "
            "time_sec, confidence, image_path, severity) VALUES (?,?,?,?,?,?,?,?)",
            (inspection_id, d.get("defect_no"), d.get("defect_type"), d.get("frame_no", 0),
             d.get("time_sec", 0.0), d.get("confidence", 0.0), d.get("image_path", ""),
             d.get("severity", "Low")))


def load_history(username=None):
    if username:
        return db_query("SELECT * FROM inspections WHERE username = ? ORDER BY id DESC", (username,))
    return db_query("SELECT * FROM inspections ORDER BY id DESC")


def load_defects(inspection_id):
    df = db_query("SELECT * FROM defects WHERE inspection_id = ? ORDER BY defect_no", (int(inspection_id),))
    return df.to_dict("records")


# ============================================================
# DETECTION HELPERS
# ============================================================

def extract_detections(result):
    """List of (class_id, confidence, (x1, y1, x2, y2))."""
    dets = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, conf):
            dets.append((int(c), float(s), tuple(float(v) for v in b)))
    return dets


def count_by_class(dets):
    counts = {k: 0 for k in CLASS_NAMES}
    for c, _, _ in dets:
        counts[c] = counts.get(c, 0) + 1
    return counts


def severity_from_box(box, image_shape):
    """
    Estimated severity based on the detected box's percentage of image area.
    This is a demo heuristic, not an engineering-grade defect severity assessment.
    Tune these thresholds with an inspector/quality engineer before production use.
    """
    height, width = image_shape[:2]
    image_area = max(1, int(height) * int(width))
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = box_area / image_area
    if area_ratio >= 0.05:
        return "High"
    if area_ratio >= 0.01:
        return "Medium"
    return "Low"


def highest_severity(values):
    order = {"Low": 1, "Medium": 2, "High": 3}
    values = [v for v in values if v in order]
    return max(values, key=lambda v: order[v]) if values else "None"


def log_alert(inspection_id, part_id, result, severity, message):
    if result != "FAIL":
        return
    db_exec(
        "INSERT INTO alert_logs (inspection_id, part_id, alert_type, severity, message, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (inspection_id, part_id, "DEFECT_DETECTED", severity, message,
         time.strftime("%Y-%m-%d %H:%M:%S"))
    )


def show_fail_alert(part_id, severity, message="Defect(s) detected. Please inspect this part."):
    st.error(f"🚨 QUALITY ALERT — PART {part_id or 'ID NOT PROVIDED'}: FAIL | "
             f"Estimated severity: {severity}. {message}")


def draw_detections(img, dets, ids=None):
    out = img.copy()
    for i, (c, s, b) in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in b]
        color = CLASS_COLORS.get(c, (0, 255, 0))
        label = f"{CLASS_NAMES.get(c, c)} {s:.2f}"
        if ids is not None and ids[i] is not None:
            label += f" #{ids[i]}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_banner(img, text, fail):
    out = img.copy()
    color = (0, 0, 200) if fail else (0, 150, 0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), color, -1)
    cv2.putText(out, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ============================================================
# DEFECT TRACKER (counts UNIQUE defects + keeps best snapshot)
# ============================================================

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class DefectTracker:
    """
    Counts each distinct defect once. When a frame is passed to update(),
    the clearest (highest-confidence) frame of every defect is stored as
    JPEG bytes so it can be shown as evidence later.
    """

    def __init__(self, min_hits=2, iou_thr=0.3, max_age=8):
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = {}
        self.registry = {}
        self.next_id = 1
        self.counts = {k: 0 for k in CLASS_NAMES}

    @property
    def total(self):
        return sum(self.counts.values())

    def _drop_unconfirmed_snaps(self, tracks):
        for t in tracks:
            if t["hits"] < self.min_hits:
                t["snap"] = None          # free memory for false alarms

    def reset_scene(self):
        self._drop_unconfirmed_snaps(self.tracks.values())
        self.tracks = {}

    def reset_all(self):
        self.__init__(self.min_hits, self.iou_thr, self.max_age)

    def update(self, dets, frame_idx, frame=None):
        pairs = []
        for di, (c, _, b) in enumerate(dets):
            for tid, t in self.tracks.items():
                if t["cls"] == c:
                    s = iou(b, t["box"])
                    if s >= self.iou_thr:
                        pairs.append((s, di, tid))
        pairs.sort(reverse=True)

        assigned = [None] * len(dets)
        used = set()
        for _, di, tid in pairs:
            if assigned[di] is None and tid not in used:
                assigned[di] = tid
                used.add(tid)

        for di, (c, conf, b) in enumerate(dets):
            tid = assigned[di]
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                t = {"cls": c, "box": b, "hits": 0, "last": frame_idx, "no": None,
                     "max_conf": 0.0, "snap": None, "snap_box": None,
                     "snap_frame": None, "snap_conf": 0.0}
                self.tracks[tid] = t
                self.registry[tid] = t
                assigned[di] = tid
            t = self.tracks[tid]
            t["box"] = b
            t["hits"] += 1
            t["last"] = frame_idx

            if frame is not None and (t["snap"] is None or conf > t["max_conf"] + 0.02):
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    t["snap"] = buf.tobytes()
                    t["snap_box"] = b
                    t["snap_frame"] = frame_idx
                    t["snap_conf"] = conf
            t["max_conf"] = max(t["max_conf"], conf)

            if t["hits"] == self.min_hits:
                self.counts[c] = self.counts.get(c, 0) + 1
                t["no"] = self.total        # display number of this defect

        nos = [self.registry[tid]["no"] for tid in assigned]

        for tid in [k for k, t in self.tracks.items() if frame_idx - t["last"] > self.max_age]:
            self._drop_unconfirmed_snaps([self.tracks[tid]])
            del self.tracks[tid]
        return nos

    def confirmed_tracks(self):
        ts = [t for t in self.registry.values() if t["hits"] >= self.min_hits and t["no"] is not None]
        return sorted(ts, key=lambda t: t["no"])


def render_snapshot(img, t, fps):
    """Evidence image: the defect box highlighted + frame/time banner."""
    out = draw_detections(img, [(t["cls"], t["snap_conf"], t["snap_box"])], [t["no"]])
    x1, y1, x2, y2 = [int(v) for v in t["snap_box"]]
    cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), CLASS_COLORS.get(t["cls"], (0, 255, 0)), 2)
    text = f"Defect #{t['no']}  |  Frame {t['snap_frame']}  |  {t['snap_frame'] / fps:.2f}s"
    return draw_banner(out, text, True)


# ============================================================
# IMAGE INFERENCE
# ============================================================

def run_image_inference(image, confidence=0.25):
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    result = model.predict(source=bgr, conf=confidence, imgsz=640, verbose=False)[0]
    dets = extract_detections(result)
    counts = count_by_class(dets)
    total = sum(counts.values())
    defect_rows = []
    for idx, (class_id, conf, box) in enumerate(dets, start=1):
        defect_rows.append({
            "defect_no": idx,
            "defect_type": CLASS_NAMES.get(class_id, str(class_id)),
            "frame_no": 0,
            "time_sec": 0.0,
            "confidence": round(conf, 4),
            "image_path": "",
            "severity": severity_from_box(box, bgr.shape),
        })
    return {
        "annotated": draw_detections(bgr, dets),
        "counts": counts,
        "total": total,
        "types": sum(1 for v in counts.values() if v > 0),
        "result": "FAIL" if total > 0 else "PASS",
        "avg_conf": float(np.mean([d[1] for d in dets])) if dets else 0.0,
        "defects": defect_rows,
        "severity": highest_severity([d["severity"] for d in defect_rows]),
    }


# ============================================================
# VIDEO PROCESSING
# ============================================================

def reencode_h264(src, dst):
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    try:
        subprocess.run(
            [exe, "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", dst],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(dst)
    except Exception:
        return False


def process_uploaded_video(input_path, output_path, snap_dir, confidence=0.25,
                           min_hits=2, scene_threshold=25.0, progress_cb=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    raw_path = output_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    tracker = DefectTracker(min_hits=min_hits)
    frame_count = detection_frames = scene_changes = 0
    all_conf = []
    prev_small = None
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        if prev_small is not None and float(np.mean(np.abs(small - prev_small))) > scene_threshold:
            tracker.reset_scene()
            scene_changes += 1
        prev_small = small

        result = model.predict(source=frame, conf=confidence, imgsz=640, verbose=False)[0]
        dets = extract_detections(result)
        if dets:
            detection_frames += 1
            all_conf.extend(d[1] for d in dets)

        nos = tracker.update(dets, frame_count, frame=frame)
        annotated = draw_detections(frame, dets, nos)
        c = tracker.counts
        text = (f"Frame {frame_count} | Defects: {tracker.total} "
                f"(Crack {c.get(0, 0)}, Porosity {c.get(1, 0)}, Scratch {c.get(2, 0)})")
        writer.write(draw_banner(annotated, text, tracker.total > 0))

        if progress_cb and total_frames > 0 and frame_count % 5 == 0:
            progress_cb(min(frame_count / total_frames, 1.0))

    cap.release()
    writer.release()
    processing_time = time.time() - start

    if not reencode_h264(raw_path, output_path):
        shutil.move(raw_path, output_path)
    elif os.path.exists(raw_path):
        os.remove(raw_path)

    # ---- save one evidence image per confirmed defect ----
    os.makedirs(snap_dir, exist_ok=True)
    defects = []
    for t in tracker.confirmed_tracks():
        path = os.path.join(snap_dir, f"defect_{t['no']:02d}.jpg")
        if t["snap"] is not None:
            img = cv2.imdecode(np.frombuffer(t["snap"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(path, render_snapshot(img, t, fps))
        defects.append({
            "defect_no": t["no"],
            "defect_type": CLASS_NAMES.get(t["cls"], str(t["cls"])),
            "frame_no": int(t["snap_frame"] or 0),
            "time_sec": round((t["snap_frame"] or 0) / fps, 2),
            "confidence": round(t["snap_conf"], 3),
            "image_path": path,
            "severity": severity_from_box(t["snap_box"], (height, width, 3)),
        })

    counts = dict(tracker.counts)
    total = tracker.total
    return {
        "result": "FAIL" if total > 0 else "PASS",
        "counts": counts,
        "total_defects": total,
        "defect_types": sum(1 for v in counts.values() if v > 0),
        "total_frames": frame_count,
        "detection_frames": detection_frames,
        "scene_changes": scene_changes,
        "original_fps": fps,
        "effective_fps": frame_count / processing_time if processing_time > 0 else 0.0,
        "processing_time": processing_time,
        "average_confidence": float(np.mean(all_conf)) if all_conf else 0.0,
        "defects": defects,
        "output_path": output_path,
    }


# ============================================================
# WEBCAM PROCESSOR
# ============================================================

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.confidence = 0.25
        self.frame_count = 0
        self.fps_frames = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.last_dets = []
        self.last_ids = []
        self.current_counts = {k: 0 for k in CLASS_NAMES}
        self.current_severity = "None"
        self.session_severity = "None"
        self.tracker = DefectTracker(min_hits=2, max_age=15)
        self.prev_small = None
        self.lock = threading.Lock()
        self.reset_requested = False

    def snapshot(self):
        with self.lock:
            return {
                "current": dict(self.current_counts),
                "session": dict(self.tracker.counts),
                "session_total": self.tracker.total,
                "fps": self.fps,
                "current_severity": self.current_severity,
                "session_severity": self.session_severity,
            }

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        self.fps_frames += 1

        h, w = image.shape[:2]
        if w > 720:
            image = cv2.resize(image, (720, int(h * 720 / w)))

        if self.frame_count % 3 == 0 and self.lock.acquire(blocking=False):
            try:
                if self.reset_requested:
                    self.tracker.reset_all()
                    self.session_severity = "None"
                    self.reset_requested = False

                small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                if self.prev_small is not None and float(np.mean(np.abs(small - self.prev_small))) > 40:
                    self.tracker.reset_scene()
                self.prev_small = small

                result = model.predict(source=image, conf=self.confidence, imgsz=512, verbose=False)[0]
                dets = extract_detections(result)
                self.last_ids = self.tracker.update(dets, self.frame_count)
                self.last_dets = dets
                self.current_counts = count_by_class(dets)
                levels = [severity_from_box(box, image.shape) for _, _, box in dets]
                self.current_severity = highest_severity(levels)
                if levels:
                    self.session_severity = highest_severity(
                        [self.session_severity, *levels]
                    )
            except Exception:
                pass
            finally:
                self.lock.release()

        output = draw_detections(image, self.last_dets, self.last_ids)
        fail = sum(self.current_counts.values()) > 0
        output = draw_banner(output, "FAIL" if fail else "PASS", fail)

        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.fps_frames / (now - self.last_time)
            self.fps_frames = 0
            self.last_time = now

        return av.VideoFrame.from_ndarray(output, format="bgr24")


# ============================================================
# LOGIN / REGISTER
# ============================================================

def login_page():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown(
            '<div class="login-brand"><div class="login-logo">🔍</div>'
            '<div class="login-title">Metal Part Inspection</div>'
            '<div class="login-sub">AI-powered defect detection for metal parts</div>'
            '<div><span class="chip">Crack</span><span class="chip">Porosity</span>'
            '<span class="chip">Scratch</span></div>'
            '<div><span class="chip alt">Image</span><span class="chip alt">Batch</span>'
            '<span class="chip alt">Video</span><span class="chip alt">Live Camera</span></div></div>',
            unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", type="primary", use_container_width=True):
                role = authenticate(username.strip(), password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.session_state.role = role
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_register:
            new_user = st.text_input("Choose a username", key="reg_user")
            new_pass = st.text_input("Choose a password (min 6 characters)", type="password", key="reg_pass")
            confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
            if st.button("Create Account", type="primary", use_container_width=True):
                new_user = new_user.strip()
                if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", new_user):
                    st.error("Username must be 3–30 characters: letters, numbers, underscore.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pass != confirm:
                    st.error("Passwords do not match.")
                elif create_user(new_user, new_pass, "user"):
                    st.success("Account created. You can now log in from the Login tab.")
                else:
                    st.error("That username is already taken.")


def account_page():
    page_header("👤", "My Account", "Manage your profile and password")
    st.write(f"**Username:** {st.session_state.username}  |  **Role:** {st.session_state.role}")
    st.subheader("Change Password")
    old = st.text_input("Current password", type="password")
    new = st.text_input("New password (min 6 characters)", type="password")
    confirm = st.text_input("Confirm new password", type="password")
    if st.button("Update Password", type="primary", use_container_width=True):
        if not authenticate(st.session_state.username, old):
            st.error("Current password is wrong.")
        elif len(new) < 6:
            st.error("New password must be at least 6 characters.")
        elif new != confirm:
            st.error("Passwords do not match.")
        else:
            pw_hash, salt = hash_password(new)
            db_exec("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                    (pw_hash, salt, st.session_state.username))
            st.success("Password updated.")


# ============================================================
# SHARED UI PIECES
# ============================================================

def counts_metrics(counts, total, types):
    stat_cards([
        ("🧮", "Total Defects", total, "#dbeafe"),
        ("💥", "Crack", counts.get(0, 0), "#fee2e2"),
        ("🕳️", "Porosity", counts.get(1, 0), "#ffedd5"),
        ("✏️", "Scratch", counts.get(2, 0), "#ede9fe"),
        ("🏷️", "Defect Types", types, "#dcfce7"),
    ])


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, p, label, input, textarea, button { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #f6f8fc 0%, #eef2f9 100%); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #0f172a; font-weight: 700; letter-spacing: -0.01em; }

/* ---------- hero header ---------- */
.hero { display: flex; align-items: center; gap: 18px; padding: 22px 28px; border-radius: 20px;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #06b6d4 100%);
        box-shadow: 0 12px 32px rgba(37, 99, 235, .28); margin-bottom: 24px; }
.hero-icon { font-size: 2.1rem; width: 64px; height: 64px; border-radius: 18px; flex-shrink: 0;
             background: rgba(255,255,255,.18); display: flex; align-items: center; justify-content: center; }
.hero-title { font-size: 1.7rem; font-weight: 800; color: #ffffff; line-height: 1.2; }
.hero-sub { margin-top: 4px; font-size: .98rem; color: #dbeafe; }

/* ---------- stat cards ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 6px 0 22px; }
.stat-card { background: #ffffff; border: 1px solid #eef1f6; border-radius: 16px; padding: 16px 18px;
             display: flex; align-items: center; gap: 14px; box-shadow: 0 2px 12px rgba(15,23,42,.06);
             transition: transform .18s ease, box-shadow .18s ease; }
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(15,23,42,.11); }
.stat-icon { width: 46px; height: 46px; border-radius: 13px; display: flex; align-items: center;
             justify-content: center; font-size: 1.35rem; flex-shrink: 0; }
.stat-value { font-size: 1.65rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
.stat-label { font-size: .8rem; color: #64748b; font-weight: 600; margin-top: 2px; }

/* ---------- feature cards ---------- */
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-top: 8px; }
.feature-card { background: #ffffff; border: 1px solid #eef1f6; border-radius: 16px; padding: 20px;
                box-shadow: 0 2px 12px rgba(15,23,42,.06); transition: transform .18s ease, box-shadow .18s ease; }
.feature-card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(15,23,42,.11); }
.feature-icon { font-size: 1.9rem; }
.feature-title { margin: 8px 0 4px; font-weight: 700; color: #0f172a; font-size: 1.02rem; }
.feature-text { color: #64748b; font-size: .9rem; }

/* ---------- PASS / FAIL result card ---------- */
.result-card { display: flex; align-items: center; gap: 20px; border-radius: 20px; padding: 22px 28px;
               margin: 10px 0 22px; color: #fff; box-shadow: 0 14px 32px rgba(15,23,42,.20); }
.result-icon { font-size: 2.3rem; width: 68px; height: 68px; border-radius: 50%; flex-shrink: 0;
               background: rgba(255,255,255,.22); display: flex; align-items: center; justify-content: center; }
.result-title { font-size: 2rem; font-weight: 800; letter-spacing: .5px; color: #fff; line-height: 1.1; }
.result-msg { font-size: 1.05rem; opacity: .96; color: #fff; margin-top: 2px; }
.result-pass { background: linear-gradient(135deg, #15803d, #22c55e); }
.result-fail { background: linear-gradient(135deg, #b91c1c, #f43f5e); }

/* ---------- login ---------- */
.login-brand { text-align: center; margin: 2.2rem 0 1.4rem; }
.login-logo { font-size: 2.6rem; width: 84px; height: 84px; margin: 0 auto 12px; border-radius: 24px;
              background: linear-gradient(135deg, #2563eb, #06b6d4); display: flex; align-items: center;
              justify-content: center; box-shadow: 0 12px 28px rgba(37,99,235,.35); }
.login-title { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #1e3a8a, #0891b2);
               -webkit-background-clip: text; background-clip: text; color: transparent; }
.login-sub { color: #64748b; margin: 4px 0 12px; }
.chip { display: inline-block; background: #e0ecff; color: #1d4ed8; padding: 4px 12px; border-radius: 999px;
        font-size: .78rem; font-weight: 700; margin: 3px; }
.chip.alt { background: #e6fbf6; color: #0f766e; }
.stTabs { background: #ffffff; padding: 18px 22px 22px; border-radius: 20px; border: 1px solid #eef1f6;
          box-shadow: 0 14px 40px rgba(15,23,42,.10); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #14213d 100%); border-right: 0; }
section[data-testid="stSidebar"] * { color: #cbd5e1; }
section[data-testid="stSidebar"] .brand { display: flex; align-items: center; gap: 12px; padding: 6px 4px 14px; }
section[data-testid="stSidebar"] .brand-logo { width: 44px; height: 44px; border-radius: 13px; font-size: 1.4rem;
        background: linear-gradient(135deg, #2563eb, #06b6d4); display: flex; align-items: center; justify-content: center; }
section[data-testid="stSidebar"] .brand-name { color: #ffffff; font-weight: 800; font-size: 1.05rem; line-height: 1.2; }
section[data-testid="stSidebar"] .brand-sub { color: #94a3b8; font-size: .75rem; }
section[data-testid="stSidebar"] .user-chip { display: flex; align-items: center; gap: 12px; padding: 12px;
        border-radius: 14px; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.10); margin-bottom: 14px; }
section[data-testid="stSidebar"] .avatar { width: 38px; height: 38px; border-radius: 50%; color: #ffffff; font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #06b6d4); display: flex; align-items: center; justify-content: center; }
section[data-testid="stSidebar"] .uname { color: #ffffff; font-weight: 700; line-height: 1.2; }
section[data-testid="stSidebar"] .role-badge { display: inline-block; font-size: .68rem; font-weight: 700; padding: 2px 9px;
        border-radius: 999px; text-transform: uppercase; letter-spacing: .5px; margin-top: 3px; }
section[data-testid="stSidebar"] .role-admin { background: #fde68a; color: #854d0e; }
section[data-testid="stSidebar"] .role-user { background: #bfdbfe; color: #1e40af; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { padding: 10px 14px; border-radius: 12px;
        transition: background .15s ease; width: 100%; cursor: pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255,255,255,.09); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #2563eb, #06b6d4); box-shadow: 0 6px 16px rgba(37,99,235,.35); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * { color: #ffffff; font-weight: 700; }
section[data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); }
section[data-testid="stSidebar"] .stButton > button * { color: #ffffff; }
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(244,63,94,.25); border-color: #f43f5e; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
        border-radius: 12px; font-weight: 600; padding: .6rem 1rem; border: 1px solid #d6deee; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 8px 18px rgba(37,99,235,.18); border-color: #2563eb; color: #2563eb; }
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2563eb, #06b6d4); color: #ffffff; border: 0; }
button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * { color: #ffffff; }
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        color: #ffffff; box-shadow: 0 10px 22px rgba(37,99,235,.40); }

/* ---------- inputs, uploader, misc ---------- */
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 12px; }
div[data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #93b4f6; border-radius: 16px; background: #f8faff; }
div[data-testid="stExpander"] { border-radius: 14px; border: 1px solid #e5eaf3; background: #ffffff; }
div[data-testid="stAlert"] { border-radius: 14px; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #e5eaf3; }
div[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 4px 16px rgba(15,23,42,.10); }
div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #eef1f6; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 2px 10px rgba(15,23,42,.05); }
div[data-testid="stMetricValue"] { font-weight: 800; color: #0f172a; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #eef2f9; padding: 6px; border-radius: 14px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 18px; height: auto; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #ffffff; color: #2563eb; box-shadow: 0 2px 8px rgba(15,23,42,.08); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
</style>
""", unsafe_allow_html=True)


def page_header(icon, title, subtitle=""):
    st.markdown(
        f'<div class="hero"><div class="hero-icon">{icon}</div><div>'
        f'<div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True)


def stat_cards(items):
    """items: list of (icon, label, value, icon_background_color)."""
    cards = "".join(
        f'<div class="stat-card"><div class="stat-icon" style="background:{bg};">{icon}</div>'
        f'<div><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div></div>'
        for icon, label, value, bg in items)
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)


def result_card(result, total, where=""):
    """Big green PASS / red FAIL card."""
    if result == "FAIL":
        cls, icon, msg = "result-fail", "❌", f"{total} defect(s) detected{where}."
    else:
        cls, icon, msg = "result-pass", "✅", f"No defects found{where} 🎉"
    st.markdown(
        f'<div class="result-card {cls}"><div class="result-icon">{icon}</div><div>'
        f'<div class="result-title">{result}</div><div class="result-msg">{msg}</div></div></div>',
        unsafe_allow_html=True)


def show_table(df, result_col="result"):
    """Dataframe with PASS (green) / FAIL (red) colored cells."""
    def color(v):
        if v == "FAIL":
            return "background-color:#fde8e8;color:#b42318;font-weight:600"
        if v == "PASS":
            return "background-color:#e7f6ec;color:#067647;font-weight:600"
        return ""
    try:
        if result_col in df.columns:
            styler = df.style
            apply_fn = getattr(styler, "map", None) or styler.applymap
            st.dataframe(apply_fn(color, subset=[result_col]),
                         use_container_width=True, hide_index=True)
            return
    except Exception:
        pass
    st.dataframe(df, use_container_width=True, hide_index=True)


def sidebar_brand(username, role):
    initial = escape(username[:1].upper() or "?")
    st.markdown(
        '<div class="brand"><div class="brand-logo">🔍</div><div>'
        '<div class="brand-name">Metal Inspection</div>'
        '<div class="brand-sub">Defect Detection System</div></div></div>'
        f'<div class="user-chip"><div class="avatar">{initial}</div><div>'
        f'<div class="uname">{escape(username)}</div>'
        f'<span class="role-badge role-{escape(role)}">{escape(role)}</span></div></div>',
        unsafe_allow_html=True)


def show_defect_gallery(defects, per_row=3):
    for i in range(0, len(defects), per_row):
        cols = st.columns(per_row)
        for col, d in zip(cols, defects[i:i + per_row]):
            with col:
                if d["image_path"] and os.path.exists(d["image_path"]):
                    st.image(d["image_path"], use_container_width=True)
                st.caption(f"**Defect #{d['defect_no']} — {d['defect_type']}**  \n"
                           f"Time {d['time_sec']:.2f}s · Frame {d['frame_no']} · "
                           f"Confidence {d['confidence'] * 100:.0f}%")


def zip_defects(defects):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for d in defects:
            if d["image_path"] and os.path.exists(d["image_path"]):
                z.write(d["image_path"], os.path.basename(d["image_path"]))
    return buf.getvalue()


# ============================================================
# SINGLE IMAGE
# ============================================================

def single_image_page():
    page_header("🖼️", "Single Image Inspection", "Upload one photo and get an instant PASS / FAIL result")
    part_id = st.text_input("Part ID", placeholder="e.g. PART-0001", key="single_part_id").strip()
    confidence = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                key="single_image",
                                help="Use a clear, well-lit photo of the metal part (JPG, PNG, BMP or WEBP).")
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Inspect Image", type="primary", use_container_width=True):
        start = time.time()
        with st.spinner("Inspecting image..."):
            res = run_image_inference(image, confidence)
        pt = time.time() - start

        if not part_id:
            st.warning("Part ID was left blank. The inspection will be saved as ID NOT PROVIDED.")
            part_id = "ID-NOT-PROVIDED"

        st.subheader("Inspection Result")
        result_card(res["result"], res["total"])
        st.write(f"**Part ID:** {part_id}  |  **Estimated severity:** {res['severity']}")
        if res["result"] == "FAIL":
            show_fail_alert(part_id, res["severity"])
        else:
            st.success("✅ No defects detected. No alert generated.")

        counts_metrics(res["counts"], res["total"], res["types"])
        st.write(f"**Processing time:** {pt:.2f} sec  |  "
                 f"**Average confidence:** {res['avg_conf'] * 100:.1f}%")
        st.caption("Severity is currently estimated from bounding-box area and must be calibrated with a quality inspector.")
        st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB),
                 caption="Inspection Result", use_container_width=True)

        insp_id = save_history(st.session_state.username, "Single Image", uploaded.name,
                               res["result"], res["counts"], pt, part_id, res["severity"])
        save_defects(insp_id, res["defects"])
        if res["result"] == "FAIL":
            log_alert(insp_id, part_id, res["result"], res["severity"],
                      f"{res['total']} defect(s) detected in {uploaded.name}")


# ============================================================
# BATCH
# ============================================================

def batch_page():
    page_header("📁", "Batch Inspection", "Inspect many images at once and download a report")
    part_prefix = st.text_input("Part ID prefix for batch", value="PART", key="batch_part_prefix",
                                help="Each image gets a unique ID such as PART-001, PART-002.")
    confidence = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key="batch_conf")
    files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True, key="batch_files",
                             help="Select many images at once. Each image gets its own PASS/FAIL result.")

    if files and st.button("Inspect Batch", type="primary", use_container_width=True):
        rows, gallery = [], []
        bar = st.progress(0.0, text="Inspecting...")
        for i, f in enumerate(files):
            start = time.time()
            image = Image.open(f).convert("RGB")
            res = run_image_inference(image, confidence)
            pt = time.time() - start
            part_id = f"{part_prefix.strip() or 'PART'}-{i + 1:03d}"
            rows.append({
                "Part ID": part_id, "File": f.name, "Result": res["result"],
                "Crack": res["counts"][0], "Porosity": res["counts"][1],
                "Scratch": res["counts"][2], "Total Defects": res["total"],
                "Severity": res["severity"], "Defect Types": res["types"],
                "Processing Time": round(pt, 2),
            })
            gallery.append((part_id, f.name, res["result"], cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB)))
            insp_id = save_history(st.session_state.username, "Batch", f.name,
                                   res["result"], res["counts"], pt, part_id, res["severity"])
            save_defects(insp_id, res["defects"])
            if res["result"] == "FAIL":
                log_alert(insp_id, part_id, res["result"], res["severity"],
                          f"{res['total']} defect(s) detected in {f.name}")
            bar.progress((i + 1) / len(files), text=f"Inspected {i + 1}/{len(files)}")
        bar.empty()
        st.session_state.batch_rows = rows
        st.session_state.batch_gallery = gallery

    rows = st.session_state.get("batch_rows")
    if rows:
        df = pd.DataFrame(rows)
        stat_cards([
            ("🖼️", "Images", len(df), "#dbeafe"),
            ("✅", "PASS", int((df["Result"] == "PASS").sum()), "#dcfce7"),
            ("❌", "FAIL", int((df["Result"] == "FAIL").sum()), "#fee2e2"),
            ("💥", "Total Defects", int(df["Total Defects"].sum()), "#ffedd5"),
        ])
        show_table(df, "Result")
        st.download_button("Download Batch Report (CSV)", df.to_csv(index=False).encode(),
                           "batch_report.csv", "text/csv", use_container_width=True)

        with st.expander("View annotated images"):
            for part_id, name, result, img in st.session_state.get("batch_gallery", []):
                st.image(img, caption=f"{part_id} — {name} — {result}", use_container_width=True)


# ============================================================
# UPLOADED VIDEO
# ============================================================

def uploaded_video_page():
    page_header("🎥", "Video Inspection",
                "Every frame is checked. Each defect is counted once and its clearest frame is saved as evidence.")

    part_id_input = st.text_input("Part ID", placeholder="e.g. PART-0001", key="video_part_id").strip()
    confidence = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key="video_conf")
    with st.expander("Advanced settings"):
        min_hits = st.slider("Frames a defect must appear in to be counted", 1, 10, 2,
                             help="1 = count anything seen once. Higher = fewer false alarms.")
        scene_thr = st.slider("Scene-change sensitivity (lower = more sensitive)", 5, 80, 25,
                              help="How different two frames must be to treat it as a new part/image.")

    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"], key="video_upload",
                                help="MP4, AVI, MOV or MKV. Every frame is checked, so long videos take more time.")
    if not uploaded:
        return

    st.video(uploaded)

    if st.button("Inspect Video", type="primary", use_container_width=True):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_DIR) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_input = tmp.name

        run_id = time.strftime("%Y%m%d_%H%M%S")
        output_video = os.path.join(OUTPUT_DIR, f"processed_{run_id}.mp4")
        snap_dir = os.path.join(DEFECT_DIR, f"{st.session_state.username}_{run_id}")

        bar = st.progress(0.0, text="Processing every frame...")
        try:
            res = process_uploaded_video(
                temp_input, output_video, snap_dir, confidence, min_hits, float(scene_thr),
                progress_cb=lambda p: bar.progress(p, text="Processing every frame..."))
        except Exception as e:
            bar.empty()
            st.error(f"Video processing failed: {e}")
            return
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)
        bar.empty()

        part_id = part_id_input or "ID-NOT-PROVIDED"
        res["part_id"] = part_id
        res["severity"] = highest_severity([d.get("severity", "Low") for d in res["defects"]])
        res["file_name"] = uploaded.name
        insp_id = save_history(st.session_state.username, "Uploaded Video", uploaded.name,
                               res["result"], res["counts"], res["processing_time"],
                               part_id, res["severity"])
        save_defects(insp_id, res["defects"])
        if res["result"] == "FAIL":
            log_alert(insp_id, part_id, res["result"], res["severity"],
                      f"{res['total_defects']} defect(s) detected in video {uploaded.name}")
        res["inspection_id"] = insp_id
        st.session_state.video_result = res

    res = st.session_state.get("video_result")
    if not res or res.get("file_name") != uploaded.name:
        return

    st.subheader("Inspection Result")
    result_card(res["result"], res["total_defects"], " in the video")
    st.write(f"**Part ID:** {res.get('part_id', 'ID-NOT-PROVIDED')}  |  "
             f"**Estimated severity:** {res.get('severity', 'None')}")
    if res["result"] == "FAIL":
        show_fail_alert(res.get("part_id", ""), res.get("severity", "None"),
                        "Defect(s) were detected in the uploaded video.")
    else:
        st.success("✅ No defects detected. No alert generated.")

    counts_metrics(res["counts"], res["total_defects"], res["defect_types"])

    stat_cards([
        ("🎞️", "Total Frames", res["total_frames"], "#dbeafe"),
        ("🎯", "Frames With Detections", res["detection_frames"], "#fee2e2"),
        ("🔀", "Scene Changes", res["scene_changes"], "#ede9fe"),
        ("📈", "Avg Confidence", f"{res['average_confidence'] * 100:.1f}%", "#dcfce7"),
    ])

    st.write(f"**Video FPS:** {res['original_fps']:.2f}  |  "
             f"**Processing speed:** {res['effective_fps']:.2f} FPS  |  "
             f"**Processing time:** {res['processing_time']:.2f} sec")

    if res["defects"]:
        st.subheader("🖼️ Defect Frames (evidence)")
        show_defect_gallery(res["defects"])
        st.download_button("Download All Defect Images (ZIP)", zip_defects(res["defects"]),
                           f"defect_frames_{res['inspection_id']}.zip", "application/zip",
                           use_container_width=True)

        st.subheader("Defect Log")
        log = pd.DataFrame(res["defects"]).drop(columns=["image_path"])
        log = log.rename(columns={
            "defect_no": "Defect No", "defect_type": "Type", "frame_no": "Frame",
            "time_sec": "Time (s)", "confidence": "Confidence", "severity": "Severity"
        })
        st.dataframe(log, use_container_width=True, hide_index=True)

    if os.path.exists(res["output_path"]):
        st.subheader("Processed Video")
        with open(res["output_path"], "rb") as f:
            data = f.read()
        st.video(data)
        st.download_button("Download Processed Video", data, "processed_video.mp4",
                           "video/mp4", use_container_width=True)


# ============================================================
# LIVE CAMERA
# ============================================================

def webcam_page():
    page_header("📷", "Live Camera Inspection", "Real-time defect detection from your webcam")
    st.warning("For webcam testing, use http://localhost:8501 and allow camera permission.")
    part_id = st.text_input("Part ID for this camera session", placeholder="e.g. PART-0001",
                            key="webcam_part_id").strip()
    confidence = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key="webcam_conf")
    st.write("YOLO runs on every 3rd frame to keep the stream responsive. "
             "Session counts include each distinct defect once.")

    ctx = webrtc_streamer(
        key="metal-inspection-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}},
            "audio": False,
        },
        async_processing=True,
    )

    if not ctx.video_processor:
        return

    ctx.video_processor.confidence = confidence

    b1, b2 = st.columns(2)
    if b1.button("Reset session counters", use_container_width=True):
        ctx.video_processor.reset_requested = True
    if b2.button("Save session to history", use_container_width=True):
        snap = ctx.video_processor.snapshot()
        final_result = "FAIL" if snap["session_total"] > 0 else "PASS"
        part_value = part_id or "ID-NOT-PROVIDED"
        estimated_severity = snap.get("session_severity", "None")
        insp_id = save_history(st.session_state.username, "Live Camera", "webcam_session",
                               final_result, snap["session"], 0.0, part_value, estimated_severity)
        if final_result == "FAIL":
            log_alert(insp_id, part_value, final_result, estimated_severity,
                      f"{snap['session_total']} defect(s) detected during live camera session")
            show_fail_alert(part_value, estimated_severity,
                            "Defect(s) were detected during the camera session.")
        st.success("Session saved.")

    st.markdown("### Live Detection")
    holder = st.empty()
    while ctx.state.playing:
        snap = ctx.video_processor.snapshot()
        cur_total = sum(snap["current"].values())
        with holder.container():
            stat_cards([
                ("❌" if cur_total else "✅", "Live Result", "FAIL" if cur_total else "PASS",
                 "#fee2e2" if cur_total else "#dcfce7"),
                ("👁️", "In View", cur_total, "#dbeafe"),
                ("💥", "Session Crack", snap["session"].get(0, 0), "#fee2e2"),
                ("🕳️", "Session Porosity", snap["session"].get(1, 0), "#ffedd5"),
                ("✏️", "Session Scratch", snap["session"].get(2, 0), "#ede9fe"),
                ("⚠️", "Estimated Severity", snap.get("current_severity", "None"), "#ffedd5"),
                ("⚡", "FPS", f"{snap['fps']:.1f}", "#e0f2fe"),
            ])
        time.sleep(0.5)


# ============================================================
# HISTORY / DASHBOARD / USERS
# ============================================================

def history_page():
    page_header("📋", "My Inspection History", "Your past inspections and defect evidence")
    df = load_history(st.session_state.username)
    if df.empty:
        st.info("No inspection history available.")
        return

    show_table(df)
    st.download_button("Download My History (CSV)", df.to_csv(index=False).encode(),
                       "my_history.csv", "text/csv")

    st.subheader("🚨 Alert Log")
    alerts = db_query(
        "SELECT a.id AS alert_id, a.part_id, a.alert_type, a.severity, a.message, "
        "a.created_at, i.file_name, i.inspection_type "
        "FROM alert_logs a LEFT JOIN inspections i ON a.inspection_id = i.id "
        "WHERE i.username = ? OR i.username IS NULL ORDER BY a.id DESC",
        (st.session_state.username,))
    if alerts.empty:
        st.info("No alerts recorded.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
        st.download_button("Download Alert Log (CSV)", alerts.to_csv(index=False).encode(),
                           "alert_log.csv", "text/csv")

    video_df = df[(df["inspection_type"] == "Uploaded Video") & (df["total_defects"] > 0)]
    if not video_df.empty:
        st.subheader("View Defect Frames of a Past Video")
        options = {f"#{r.id} — {r.file_name} ({r.timestamp})": r.id for r in video_df.itertuples()}
        pick = st.selectbox("Select inspection", list(options.keys()))
        defects = load_defects(options[pick])
        if defects:
            show_defect_gallery(defects)
        else:
            st.info("No saved defect frames for this inspection.")


def home_page():
    page_header("👋", f"Welcome, {escape(st.session_state.username)}",
                "Check metal parts for Crack, Porosity and Scratch defects "
                "using images, videos or a live camera.")

    if authenticate(st.session_state.username, "admin123") or \
            authenticate(st.session_state.username, "user123"):
        st.warning("You are using a default password. Please change it in **Account**.")

    df = load_history(st.session_state.username)
    if df.empty:
        st.info("No inspections yet. Choose a mode from the menu on the left to start.")
    else:
        stat_cards([
            ("🧾", "My Inspections", len(df), "#dbeafe"),
            ("✅", "PASS", int((df["result"] == "PASS").sum()), "#dcfce7"),
            ("❌", "FAIL", int((df["result"] == "FAIL").sum()), "#fee2e2"),
            ("💥", "Defects Found", int(df["total_defects"].sum()), "#ffedd5"),
        ])
        st.subheader("Recent Inspections")
        show_table(df[["inspection_type", "file_name", "result", "total_defects", "timestamp"]].head(5))

    st.subheader("Quick Start")
    st.markdown(
        '<div class="feature-grid">'
        '<div class="feature-card"><div class="feature-icon">🖼️</div><div class="feature-title">Single Image</div>'
        '<div class="feature-text">Check one photo and see the defects marked.</div></div>'
        '<div class="feature-card"><div class="feature-icon">📁</div><div class="feature-title">Batch</div>'
        '<div class="feature-text">Check many photos at once and download a report.</div></div>'
        '<div class="feature-card"><div class="feature-icon">🎥</div><div class="feature-title">Video</div>'
        '<div class="feature-text">Count every defect and save its clearest frame.</div></div>'
        '<div class="feature-card"><div class="feature-icon">📷</div><div class="feature-title">Live Camera</div>'
        '<div class="feature-text">Inspect parts in real time from your webcam.</div></div>'
        '</div>', unsafe_allow_html=True)


def admin_dashboard():
    if st.session_state.role != "admin":
        st.error("Access denied. This page is for admins only.")
        return
    page_header("📊", "Admin Dashboard", "Overview of all inspections across users")
    df = load_history()
    if df.empty:
        st.info("No inspection data available.")
        return

    n_pass = int((df["result"] == "PASS").sum())
    n_fail = int((df["result"] == "FAIL").sum())
    stat_cards([
        ("🧾", "Total Inspections", len(df), "#dbeafe"),
        ("✅", "PASS", n_pass, "#dcfce7"),
        ("❌", "FAIL", n_fail, "#fee2e2"),
        ("📉", "Defect Rate", f"{n_fail / len(df) * 100:.1f}%", "#ffedd5"),
        ("👥", "Registered Users", int(db_query("SELECT COUNT(*) AS n FROM users")["n"][0]), "#ede9fe"),
    ])

    left, right = st.columns(2)
    with left:
        st.subheader("Defects by Type")
        st.bar_chart(pd.Series({"Crack": df["crack"].sum(),
                                "Porosity": df["porosity"].sum(),
                                "Scratch": df["scratch"].sum()}))
    with right:
        st.subheader("Inspections by Mode")
        st.bar_chart(df["inspection_type"].value_counts())

    st.subheader("🚨 Recent Alert Log")
    alerts = db_query(
        "SELECT id AS alert_id, part_id, alert_type, severity, message, created_at "
        "FROM alert_logs ORDER BY id DESC LIMIT 200")
    if alerts.empty:
        st.info("No alerts recorded.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)

    st.subheader("Inspection History")
    users = ["All"] + sorted(df["username"].dropna().unique().tolist())
    pick = st.selectbox("Filter by user", users)
    view = df if pick == "All" else df[df["username"] == pick]
    show_table(view)
    st.download_button("Download History (CSV)", view.to_csv(index=False).encode(),
                       "inspection_history.csv", "text/csv")


def manage_users_page():
    if st.session_state.role != "admin":
        st.error("Access denied. This page is for admins only.")
        return
    page_header("👥", "Manage Users", "Change roles or remove users")
    users = db_query(
        "SELECT u.username, u.role, u.created_at, "
        "(SELECT COUNT(*) FROM inspections i WHERE i.username = u.username) AS inspections "
        "FROM users u ORDER BY u.id")
    st.dataframe(users, use_container_width=True, hide_index=True)

    others = [u for u in users["username"] if u != st.session_state.username]
    if not others:
        st.info("No other users to manage.")
        return

    target = st.selectbox("Select user", others)
    col1, col2 = st.columns(2)
    with col1:
        new_role = st.selectbox("Role", ["user", "admin"])
        if st.button("Update Role", use_container_width=True):
            db_exec("UPDATE users SET role = ? WHERE username = ?", (new_role, target))
            st.success(f"{target} is now {new_role}.")
            st.rerun()
    with col2:
        st.write("")
        st.write("")
        if st.button("Delete User", use_container_width=True):
            db_exec("DELETE FROM users WHERE username = ?", (target,))
            st.success(f"{target} deleted (their inspection records are kept).")
            st.rerun()


# ============================================================
# SESSION + MAIN
# ============================================================

PAGE_ICONS = {
    "Home": "🏠", "Dashboard": "📊", "Manage Users": "👥", "Single Image": "🖼️",
    "Batch Inspection": "📁", "Uploaded Video": "🎥", "Live Camera": "📷",
    "History": "📋", "Account": "👤",
}

for key, default in (("logged_in", False), ("username", ""), ("role", "")):
    if key not in st.session_state:
        st.session_state[key] = default

init_db()
inject_css()

if not st.session_state.logged_in:
    login_page()
else:
    with st.sidebar:
        sidebar_brand(st.session_state.username, st.session_state.role)
        pages = ["Home", "Single Image", "Batch Inspection", "Uploaded Video",
                 "Live Camera", "History", "Account"]
        if st.session_state.role == "admin":
            pages = ["Home", "Dashboard", "Manage Users"] + pages[1:]
        menu = st.radio("Menu", pages, label_visibility="collapsed",
                        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    {
        "Home": home_page,
        "Dashboard": admin_dashboard,
        "Manage Users": manage_users_page,
        "Single Image": single_image_page,
        "Batch Inspection": batch_page,
        "Uploaded Video": uploaded_video_page,
        "Live Camera": webcam_page,
        "History": history_page,
        "Account": account_page,
    }[menu]()
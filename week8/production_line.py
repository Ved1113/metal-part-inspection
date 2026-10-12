import argparse, csv, os, time
import cv2
from pipeline import inspect_image, raise_alert, SEV_ORDER

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="camera index, video file or RTSP url")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--stride", type=int, default=2, help="inspect every Nth frame")
ap.add_argument("--part-seconds", type=float, default=3.0, help="time each part is in view")
ap.add_argument("--min-fail", type=int, default=2, help="defect frames needed to FAIL a part")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()

os.makedirs("output/line", exist_ok=True)
LOG = "output/line_log.csv"
new_log = not os.path.exists(LOG)
logf = open(LOG, "a", newline="")
w = csv.writer(logf)
if new_log:
    w.writerow(["timestamp", "part_id", "verdict", "severity", "crack", "porosity",
                "scratch", "defect_frames", "frames_checked", "image"])

cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
part_n, t_part, fi = 1, time.time(), 0
fails, checked, best, lat = 0, 0, None, []
stats = {"parts": 0, "fail": 0}


def finalize():
    global part_n, fails, checked, best
    pid = f"P{part_n:05d}"
    if fails >= a.min_fail and best:
        best["part_id"], best["verdict"] = pid, "FAIL"
        img = f"output/line/{pid}_FAIL.jpg"
        cv2.imwrite(img, best["annotated"])
        raise_alert(best)
        c = best["counts"]
        w.writerow([best["timestamp"], pid, "FAIL", best["severity"], c[0], c[1], c[2], fails, checked, img])
        stats["fail"] += 1
    else:
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), pid, "PASS", "None", 0, 0, 0, fails, checked, ""])
    logf.flush()
    stats["parts"] += 1
    print(f"{pid}: {'FAIL' if fails >= a.min_fail else 'PASS'} ({fails}/{checked} frames with defects)")
    part_n += 1; fails = checked = 0; best = None


try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % a.stride == 0:
            res = inspect_image(frame, f"P{part_n:05d}", a.conf)
            lat.append(res["processing_ms"]); checked += 1
            if res["total"]:
                fails += 1
                key = lambda r: (SEV_ORDER.index(r["severity"]), r["total"])
                if best is None or key(res) > key(best):
                    best = res
            if a.show:
                cv2.imshow("Line inspection (q to quit)", res["annotated"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if time.time() - t_part >= a.part_seconds:
            finalize(); t_part = time.time()
finally:
    if checked:
        finalize()
    cap.release(); logf.close(); cv2.destroyAllWindows()
    if lat:
        print(f"\nParts: {stats['parts']} | FAIL: {stats['fail']} | "
              f"avg latency {sum(lat)/len(lat):.1f} ms ({1000/(sum(lat)/len(lat)):.1f} FPS)")
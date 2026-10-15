import os, tempfile
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pipeline import (inspect_image, inspect_video, raise_alert, load_alerts, jpeg_b64,
                      new_part_id, warm_up)

app = FastAPI(title="Metal Part Inspection API")
VIDEO_DIR = os.path.join("output", "api_video")
os.makedirs(VIDEO_DIR, exist_ok=True)


@app.on_event("startup")
def _warm():
    warm_up()          # first request is not slow


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- IMAGE ----------------
@app.post("/inspect")
async def inspect(file: UploadFile = File(...), part_id: str = Form(""), conf: float = Form(0.25)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    res = await run_in_threadpool(inspect_image, img, part_id.strip() or None, conf)
    alerted = raise_alert(res)
    return {"part_id": res["part_id"], "verdict": res["verdict"], "severity": res["severity"],
            "counts": {"crack": res["counts"][0], "porosity": res["counts"][1], "scratch": res["counts"][2]},
            "detections": [{k: v for k, v in d.items() if k != "cls"} for d in res["detections"]],
            "processing_ms": res["processing_ms"], "timings": res["timings"], "alert": alerted,
            "annotated_jpeg_b64": jpeg_b64(res["annotated"])}


# ---------------- VIDEO ----------------
@app.post("/inspect_video")
async def inspect_video_route(file: UploadFile = File(...), part_id: str = Form(""),
                              conf: float = Form(0.25), stride: int = Form(2),
                              min_hits: int = Form(2), max_seconds: float = Form(0),
                              include_images: bool = Form(True)):
    """Upload a video. Each distinct defect is counted once; its clearest frame is returned."""
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    if suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(400, "Use an mp4, avi, mov or mkv video")
    pid = part_id.strip() or new_part_id()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await file.read(1024 * 1024):        # stream to disk, not into RAM
            tmp.write(chunk)
        path = tmp.name
    try:
        res = await run_in_threadpool(inspect_video, path, conf, stride, min_hits,
                                      25.0, 8, 0.3, max_seconds or None)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    finally:
        if os.path.exists(path):
            os.remove(path)

    # save evidence images + build the reply
    out_dir = os.path.join(VIDEO_DIR, pid)
    os.makedirs(out_dir, exist_ok=True)
    defects = []
    for d in res["defects"]:
        item = {k: v for k, v in d.items() if k not in ("annotated", "cls")}
        if d["annotated"] is not None:
            f = os.path.join(out_dir, f"defect_{d['no']:02d}.jpg")
            cv2.imwrite(f, d["annotated"])
            item["image_file"] = f
            if include_images:
                item["annotated_jpeg_b64"] = jpeg_b64(d["annotated"])
        defects.append(item)

    alert_rec = {"part_id": pid, "verdict": res["verdict"], "severity": res["severity"],
                 "counts": res["counts"], "total": res["total"], "timestamp": res["timestamp"]}
    alerted = raise_alert(alert_rec)

    return {"part_id": pid, "verdict": res["verdict"], "severity": res["severity"],
            "counts": {"crack": res["counts"][0], "porosity": res["counts"][1], "scratch": res["counts"][2]},
            "total_defects": res["total"], "defects": defects,
            "video": {"fps": res["fps"], "frames_total": res["frames_total"],
                      "frames_checked": res["frames_checked"],
                      "detection_frames": res["detection_frames"],
                      "scene_changes": res["scene_changes"]},
            "processing_s": res["processing_s"], "alert": alerted}


@app.get("/alerts")
def alerts(n: int = 20):
    return load_alerts(n)


# ---------------- DEMO PAGE ----------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html><html><head><meta charset="utf-8"><title>Metal Part Inspection</title></head>
<body style="font-family:Arial;max-width:820px;margin:30px auto">
<h2>🔍 Metal Part Inspection – Demo</h2>
<p>Choose an <b>image</b> or a <b>video</b>.</p>
<input type="file" id="f" accept="image/*,video/*"> <input id="p" placeholder="Part ID (optional)">
<button onclick="go()">Inspect</button>
<h3 id="v"></h3><div id="d"></div><img id="i" style="max-width:100%"><pre id="j"></pre>
<script>
async function go(){
  const file=document.getElementById('f').files[0]; if(!file){alert('Choose a file');return;}
  const isVid=file.type.startsWith('video');
  const fd=new FormData(); fd.append('file',file); fd.append('part_id',document.getElementById('p').value);
  document.getElementById('v').innerText='Processing... (videos take longer)';
  document.getElementById('v').style.color='black';
  document.getElementById('d').innerHTML=''; document.getElementById('i').src='';
  const resp=await fetch(isVid?'/inspect_video':'/inspect',{method:'POST',body:fd});
  const r=await resp.json();
  if(!resp.ok){document.getElementById('v').innerText='Error: '+(r.detail||resp.status);return;}
  document.getElementById('v').innerText=r.part_id+' → '+r.verdict+' ('+r.severity+')'+(r.alert?' 🚨':'');
  document.getElementById('v').style.color=r.verdict=='PASS'?'green':'crimson';
  if(isVid){
    let h='<p>Defects counted: <b>'+r.total_defects+'</b> | processing '+r.processing_s+' s</p>';
    for(const d of r.defects){
      h+='<div style="margin:12px 0"><b>#'+d.no+' '+d.type+'</b> ('+d.severity+') – '+d.time_sec+' s, frame '+d.frame+
         ', conf '+Math.round(d.confidence*100)+'%<br>'+
         (d.annotated_jpeg_b64?'<img style="max-width:100%" src="data:image/jpeg;base64,'+d.annotated_jpeg_b64+'">':'')+'</div>';
      delete d.annotated_jpeg_b64;
    }
    document.getElementById('d').innerHTML=h;
  } else {
    document.getElementById('i').src='data:image/jpeg;base64,'+r.annotated_jpeg_b64;
    delete r.annotated_jpeg_b64;
  }
  document.getElementById('j').innerText=JSON.stringify(r,null,2);
}
</script></body></html>"""


# http://localhost:8000
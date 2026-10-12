import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pipeline import inspect_image, raise_alert, load_alerts, jpeg_b64

app = FastAPI(title="Metal Part Inspection API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), part_id: str = Form(""), conf: float = Form(0.25)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    res = inspect_image(img, part_id.strip() or None, conf)
    alerted = raise_alert(res)
    return {"part_id": res["part_id"], "verdict": res["verdict"], "severity": res["severity"],
            "counts": {"crack": res["counts"][0], "porosity": res["counts"][1], "scratch": res["counts"][2]},
            "detections": [{k: v for k, v in d.items() if k != "cls"} for d in res["detections"]],
            "processing_ms": res["processing_ms"], "alert": alerted,
            "annotated_jpeg_b64": jpeg_b64(res["annotated"])}


@app.get("/alerts")
def alerts(n: int = 20):
    return load_alerts(n)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html><html><body style="font-family:Arial;max-width:760px;margin:30px auto">
<h2>🔍 Metal Part Inspection – Demo</h2>
<input type="file" id="f" accept="image/*"> <input id="p" placeholder="Part ID (optional)">
<button onclick="go()">Inspect</button>
<h3 id="v"></h3><pre id="j"></pre><img id="i" style="max-width:100%">
<script>
async function go(){
  const fd=new FormData(); fd.append('file',document.getElementById('f').files[0]);
  fd.append('part_id',document.getElementById('p').value);
  const r=await (await fetch('/inspect',{method:'POST',body:fd})).json();
  document.getElementById('v').innerText=r.part_id+' → '+r.verdict+' ('+r.severity+')'+(r.alert?' 🚨':'');
  document.getElementById('v').style.color=r.verdict=='PASS'?'green':'crimson';
  document.getElementById('i').src='data:image/jpeg;base64,'+r.annotated_jpeg_b64;
  delete r.annotated_jpeg_b64; document.getElementById('j').innerText=JSON.stringify(r,null,2);
}
</script></body></html>"""
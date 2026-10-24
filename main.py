"""Parchi Cloud OCR Server - runs the SAME RapidOCR engine as the desktop
Parchi Reader, exposed as a web API so the phone app reads photos with
identical accuracy.

POST /read  (multipart: photo file, kind=silver|gold, rate_kg=205000)
  -> { rows: [...], tilt: float, count: int }

GET /health -> { ok: true }
"""
import io
import sys
import os

# reader.py sits alongside main.py in this repo
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI(title="Parchi OCR Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load reader on first request (RapidOCR model download takes time)
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        import reader
        _reader = reader
    return _reader


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/read")
async def read_parchi(
    photo: UploadFile = File(...),
    kind: str = Form("silver"),
    rate_kg: float = Form(205000),
):
    r = get_reader()
    data = await photo.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image", "rows": []}
    rows, tilt, (W, H) = r.read_image(img)
    out = []
    for row in rows:
        c = r.compute_row(row, rate_kg)
        out.append({
            "item": c.get("item") or "",
            "gwt": c.get("gwt"),
            "net": c.get("net"),
            "tw": c.get("tw"),
            "tunch": c.get("tunch"),
            "labour_rs": c.get("labour"),
            "silver_wt": c.get("silver"),
            "flag": c.get("flag") or "",
        })
    return {"rows": out, "tilt": round(tilt, 2), "count": len(out)}

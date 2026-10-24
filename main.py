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
import math
import traceback

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

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        import reader
        _reader = reader
    return _reader


def to_num(v):
    if v is None:
        return None
    try:
        n = float(str(v).replace(',', '').strip())
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/read")
def read_parchi(
    photo: UploadFile = File(...),
    kind: str = Form("silver"),
    rate_kg: float = Form(205000),
):
    try:
        r = get_reader()
        data = photo.file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image", "rows": []}

        # Auto-scale large images so RapidOCR runs fast and fits safely in free-tier memory
        H, W = img.shape[:2]
        max_dim = 1200
        if max(H, W) > max_dim:
            scale = max_dim / float(max(H, W))
            new_w, new_h = max(1, int(W * scale)), max(1, int(H * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rows, tilt, (w_cur, h_cur) = r.read_image(img)
        out = []
        for row in rows:
            c = r.compute_row(row, rate_kg)
            out.append({
                "item": c.get("item") or "",
                "gwt": to_num(c.get("gwt")),
                "net": to_num(c.get("net")),
                "tw": to_num(c.get("tw")),
                "tunch": to_num(c.get("tunch")),
                "labour_rs": to_num(c.get("labour")),
                "silver_wt": to_num(c.get("silver")),
                "flag": c.get("flag") or "",
            })
        return {"rows": out, "tilt": round(tilt, 2), "count": len(out)}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "rows": []}
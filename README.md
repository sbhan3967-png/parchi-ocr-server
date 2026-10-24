"""Parchi OCR server - runs the SAME RapidOCR engine as the desktop Parchi Reader.

Deploy: Render / Railway / any Python host. Just `pip install -r requirements.txt`
and `uvicorn main:app --host 0.0.0.0 --port $PORT`.

POST /read  (multipart: photo file, kind=silver|gold, rate_kg=205000)
  -> { rows: [...], tilt: float, count: int }
GET /health -> { ok: true }
"""

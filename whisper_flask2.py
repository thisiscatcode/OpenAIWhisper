"""CPU-friendly Flask API backed by Faster-Whisper.

This service is useful when low-memory CPU inference matters more than loading
the fine-tuned Transformers checkpoint used by ``whisper_flask.py``.
"""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v2")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ja")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@lru_cache(maxsize=1)
def get_model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "model_loaded": get_model.cache_info().currsize > 0,
        }
    )


@app.post("/transcribe")
def transcribe_upload():
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify({"error": "multipart field 'audio' is required"}), 400

    suffix = Path(upload.filename).suffix.lower() or ".audio"
    if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}:
        return jsonify({"error": "unsupported audio format"}), 400
    started = time.perf_counter()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            upload.save(temporary)
            temporary_path = Path(temporary.name)
        segments, info = get_model().transcribe(
            str(temporary_path), language=LANGUAGE, vad_filter=True
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return jsonify(
            {
                "text": text,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

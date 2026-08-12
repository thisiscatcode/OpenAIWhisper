"""Flask API for a locally saved fine-tuned Whisper checkpoint."""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio
from flask import Flask, jsonify, request
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline,
)

app = Flask(__name__)
MODEL_DIR = Path(os.getenv("WHISPER_MODEL_DIR", "whisper-output"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@lru_cache(maxsize=1)
def get_asr_pipeline():
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
    )


def transcribe(path: Path) -> str:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    target_rate = get_asr_pipeline().feature_extractor.sampling_rate
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    result = get_asr_pipeline()(
        {"array": waveform.squeeze(0).numpy(), "sampling_rate": target_rate}
    )
    return str(result["text"]).strip()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_dir": str(MODEL_DIR)})


@app.post("/transcribe")
def transcribe_upload():
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify({"error": "multipart field 'audio' is required"}), 400

    suffix = Path(upload.filename).suffix or ".audio"
    started = time.perf_counter()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            upload.save(temporary)
            temporary_path = Path(temporary.name)
        text = transcribe(temporary_path)
        return jsonify(
            {"text": text, "elapsed_seconds": round(time.perf_counter() - started, 3)}
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

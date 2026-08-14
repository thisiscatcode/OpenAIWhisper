# Whisper Fine-Tuning and Transcription Service

Hands-on speech ML project for fine-tuning OpenAI Whisper on a custom Japanese
dataset, evaluating it with held-out Word Error Rate (WER), and serving the
result through Python transcription workflows and Flask APIs.

## What this repository demonstrates

- PyTorch and Hugging Face Transformers model training
- Custom audio/transcription dataset preparation
- Reproducible train/evaluation split created before feature extraction
- Sequence-to-sequence padding and mixed-precision GPU training
- Held-out WER evaluation and saved evaluation metrics
- Long-form audio transcription in configurable chunks
- Flask/Gunicorn inference service patterns
- Faster-Whisper CPU inference and fine-tuned Transformers inference

## Project structure

| File | Purpose |
| --- | --- |
| `finetune_whisper.py` | Fine-tunes Whisper and writes held-out WER metrics |
| `transcribe.py` | Transcribes long audio with a saved fine-tuned checkpoint |
| `whisper_flask.py` | Flask upload API for the fine-tuned Transformers checkpoint |
| `whisper_flask2.py` | CPU-friendly Flask upload API using Faster-Whisper |
| `colab_env_prepare.py` | Installs the pinned dependencies in Colab/notebooks |
| `gunicorn.conf.py` | Gunicorn configuration for long-running inference requests |

## Dataset format

Create a UTF-8 CSV with one row per audio file:

```csv
audio_file,transcription
clips/example-001.wav,これは音声の書き起こしです。
clips/example-002.wav,二番目のサンプルです。
```

Audio files are loaded with `torchaudio`, converted to mono, and resampled to
16 kHz. Keep private or licensed training audio outside this repository.

## Setup

Python 3.10+ and FFmpeg are recommended. Install dependencies in a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Fine-tuning

```bash
python finetune_whisper.py \
  --csv data/openv_ja.csv \
  --audio-root data \
  --output-dir whisper-output \
  --model openai/whisper-medium \
  --language japanese \
  --eval-size 0.1 \
  --max-steps 4000
```

The script saves the best checkpoint, processor files, and held-out evaluation
metrics under `whisper-output/`. Use a smaller model or fewer steps for a quick
smoke test. A CUDA GPU is strongly recommended for full fine-tuning.

## Transcription

```bash
python transcribe.py recording.mp3 \
  --model whisper-output \
  --language japanese \
  --chunk-seconds 30
```

## HTTP inference

Fine-tuned Transformers checkpoint:

```bash
WHISPER_MODEL_DIR=whisper-output gunicorn -c gunicorn.conf.py whisper_flask:app
curl -F "audio=@recording.mp3" http://localhost:5001/transcribe
```

CPU-friendly Faster-Whisper service:

```bash
WHISPER_MODEL=large-v2 python whisper_flask2.py
curl -F "audio=@recording.mp3" http://localhost:5002/transcribe
```

Both services expose `GET /health`. Models load lazily on the first transcription request, so process startup and orchestration checks do not allocate model memory.

Example response:

```json
{
  "text": "これは音声の書き起こしです。",
  "language": "ja",
  "language_probability": 0.9981,
  "elapsed_seconds": 2.417
}
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MODEL_DIR` | `whisper-output` | Fine-tuned Transformers checkpoint |
| `WHISPER_MODEL` | `large-v2` | Faster-Whisper model name or local path |
| `WHISPER_DEVICE` | `cpu` | Faster-Whisper device |
| `WHISPER_COMPUTE_TYPE` | `int8` | Faster-Whisper compute type |
| `WHISPER_LANGUAGE` | `ja` | Transcription language |
| `MAX_UPLOAD_MB` | `50` | Upload-size limit |
| `GUNICORN_WORKERS` | `1` | Worker count; each worker may load a model copy |

## Tests and continuous integration

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```

API tests replace the model with a lightweight fake, so CI validates upload handling, response contracts and temporary-file cleanup without downloading weights.

## Docker

The default container starts the CPU-friendly Faster-Whisper service:

```bash
docker build -t whisper-transcription-api .
docker run --rm -p 5002:5002 \
  -e WHISPER_MODEL=small \
  -e WHISPER_LANGUAGE=ja \
  whisper-transcription-api
```

Mount a local model cache or checkpoint for repeatable offline deployments.

## Evaluation status

The repository contains the complete held-out WER evaluation path but does not
publish a benchmark number yet. A future benchmark should record the dataset
version, split seed, base-model WER, fine-tuned-model WER, hardware, and training
configuration. No result should be quoted without the corresponding saved
metrics artifact.

## Engineering notes

- Training and evaluation data are separated with a deterministic seed.
- Model checkpoints, raw audio, credentials, and runtime output are ignored.
- Flask services use environment-based model configuration, upload-size and
  media-type limits, temporary-file cleanup, lazy loading, and health endpoints.
- Add authentication, rate limiting, malware scanning and production
  observability before exposing uploads publicly.
- Do not log private audio or transcripts. Define retention and deletion rules
  in the consuming product.

## License

Code in this repository is available under the MIT License. Model weights and
datasets remain subject to their respective licenses and terms.

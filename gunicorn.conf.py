"""Gunicorn defaults for model-serving workloads."""

import os

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5001")
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "600"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "60"))
accesslog = "-"
errorlog = "-"

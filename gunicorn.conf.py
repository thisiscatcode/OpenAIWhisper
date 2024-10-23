# gunicorn.conf.py

timeout = 86400  # Set to 24 hours
graceful_timeout = 86400
bind = "0.0.0.0:5001"

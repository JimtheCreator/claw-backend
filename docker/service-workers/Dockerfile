FROM python:3.13.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including Chrome for Plotly
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    supervisor \
    # Chrome dependencies for Plotly/Kaleido
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directories
COPY src/ ./src/
COPY stripe_payments/ ./stripe_payments/

# Copy Firebase credentials
COPY src/infrastructure/database/firebase/secrets/watchers-db-firebase-credentials.json /app/src/infrastructure/database/firebase/secrets/watchers-db-firebase-credentials.json

# Create supervisor directories
RUN mkdir -p /var/log/supervisor

# Create supervisor configuration for all worker services
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

# --- FIXED CELERY CONFIGURATION ---

# WORKER 1: For general I/O-bound tasks (fast, lightweight)
[program:celery_general_worker]
command=celery -A src.core.services.workers.celery_worker worker --pool=threads --concurrency=10 --loglevel=info --queues=default,telegram,broadcasts
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/celery_general_err.log
stdout_logfile=/var/log/supervisor/celery_general_out.log
environment=PYTHONPATH="/app:/app/src"

# WORKER 2: For CPU-intensive analysis tasks (slow, heavy)
[program:celery_analysis_worker]
command=celery -A src.core.services.workers.celery_worker worker --pool=processes --concurrency=2 --loglevel=info --queues=analysis
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/celery_analysis_err.log
stdout_logfile=/var/log/supervisor/celery_analysis_out.log
environment=PYTHONPATH="/app:/app/src"

[program:notification_worker]
command=python src/core/services/workers/notification_worker.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/notification_worker_err.log
stdout_logfile=/var/log/supervisor/notification_worker_out.log
environment=PYTHONPATH="/app:/app/src"

[program:ticker_service]
command=python src/core/services/workers/ticker_service.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/ticker_service_err.log
stdout_logfile=/var/log/supervisor/ticker_service_out.log
environment=PYTHONPATH="/app:/app/src"

[program:sparkline_service]
command=python src/core/services/workers/sparkline_service.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/sparkline_service_err.log
stdout_logfile=/var/log/supervisor/sparkline_service_out.log
environment=PYTHONPATH="/app:/app/src"

[program:pattern_alert_worker]
command=python src/core/services/workers/alerts/pattern_alerts/pattern_alert_worker.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/pattern_alert_worker_err.log
stdout_logfile=/var/log/supervisor/pattern_alert_worker_out.log
environment=PYTHONPATH="/app:/app/src"

[program:price_alert_worker]
command=python src/core/services/workers/alerts/price_alerts/worker.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/price_alert_worker_err.log
stdout_logfile=/var/log/supervisor/price_alert_worker_out.log
environment=PYTHONPATH="/app:/app/src"
EOF

# Health check endpoint - create a simple health check script
COPY <<EOF /app/health_check.py
#!/usr/bin/env python3
import sys
import subprocess
import json

def check_supervisor_status():
    try:
        result = subprocess.run(['supervisorctl', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            # Check if all programs are running
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'RUNNING' not in line:
                    return False
            return True
        return False
    except Exception:
        return False

if __name__ == '__main__':
    if check_supervisor_status():
        print("All worker services are running")
        sys.exit(0)
    else:
        print("Some worker services are not running")
        sys.exit(1)
EOF

RUN chmod +x /app/health_check.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python /app/health_check.py || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
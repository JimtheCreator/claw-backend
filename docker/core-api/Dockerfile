FROM python:3.13.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies, ADDING NGINX
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directories in the correct structure
COPY src/ ./src/
COPY telegram_app_handler/ ./telegram_app_handler/
COPY stripe_payments/ ./stripe_payments/

# --- NGINX SETUP ---
# Copy your custom nginx config from the correct path
COPY docker/core-api/nginx.conf /etc/nginx/sites-enabled/default

# Remove the default nginx config if it exists
RUN rm -f /etc/nginx/sites-enabled/default.dpkg-dist

# Create supervisor directories
RUN mkdir -p /var/log/supervisor

# --- SUPERVISOR CONFIG UPDATE ---
# Update supervisor config to run nginx as the entrypoint
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

# NEW: Nginx runs in the foreground and acts as the web entrypoint
[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/nginx_err.log
stdout_logfile=/var/log/supervisor/nginx_out.log

[program:main_api]
command=uvicorn src.app:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/main_api_err.log
stdout_logfile=/var/log/supervisor/main_api_out.log
environment=PYTHONPATH="/app"

[program:telegram_api]
command=python telegram_app_handler/src/server.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/telegram_err.log
stdout_logfile=/var/log/supervisor/telegram_out.log
environment=PYTHONPATH="/app"

[program:websocket_manager]
command=python src/core/services/workers/websocket_subscription_manager.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/websocket_err.log
stdout_logfile=/var/log/supervisor/websocket_out.log
environment=PYTHONPATH="/app"
EOF

# Expose the NGINX port now, not the app ports
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
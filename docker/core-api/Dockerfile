FROM python:3.13.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies INCLUDING Chrome for Plotly/Kaleido
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    supervisor \
    nginx \
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

# Copy source directories in the correct structure
COPY src/ ./src/
COPY telegram_app_handler/ ./telegram_app_handler/
COPY stripe_payments/ ./stripe_payments/
# Copy Firebase credentials
COPY src/infrastructure/database/firebase/secrets/watchers-db-firebase-credentials.json /app/src/infrastructure/database/firebase/secrets/watchers-db-firebase-credentials.json

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
environment=PYTHONPATH="/app:/app/src"

[program:telegram_api]
command=python telegram_app_handler/src/server.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/telegram_err.log
stdout_logfile=/var/log/supervisor/telegram_out.log
environment=PYTHONPATH="/app:/app/src"

[program:websocket_manager]
command=python src/core/services/workers/websocket_subscription_manager.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/websocket_err.log
stdout_logfile=/var/log/supervisor/websocket_out.log
environment=PYTHONPATH="/app:/app/src"
EOF

# Expose the NGINX port now, not the app ports
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
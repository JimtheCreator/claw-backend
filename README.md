# Claw Backend (Watchers)

Claw Backend is a high-performance, distributed financial intelligence platform designed for real-time market analysis and algorithmic pattern detection.

Unlike traditional monolithic screeners, Claw operates as a mesh of microservices, utilizing an event-driven architecture to ingest high-frequency market data, process complex quantitative models, and deliver actionable "Trader-Aware" insights via API and Telegram.

---

## 🏗 System Architecture

The system is architected as a distributed cluster of services, ensuring that heavy data science computations do not block the responsiveness of the core API.

---

## 🔌 Core Microservices

**Core API (core-api):** A FastAPI gateway that handles client connections, WebSocket streaming, and orchestrates user requests. It serves as the "brain" for routing but delegates heavy lifting.

**Data Workers (data-workers):** A scalable fleet of Celery workers. These nodes perform the CPU-intensive tasks:
- Ingesting real-time crypto/stock data.
- Running the Trader-Aware Analysis engine.
- Calculating Swing Highs/Lows, ATR, and other technical indicators.

**Notification Engine:** Decoupled handlers for delivering alerts via Telegram and Firebase, ensuring signals are pushed instantly upon pattern confirmation.

---

## 🧠 Data Science & ML-Hybrid Engine

Claw goes beyond simple if/else indicators by implementing a Trader-Aware Scoring System—a hybrid of quantitative analysis and heuristic modeling:

- **Contextual Pattern Detection:** Patterns (Harmonic, Chart, Candlestick) are not just "found"; they are validated against support/resistance zones and trend direction.
- **Multi-Factor Scoring:** Every setup is graded (0-100%) based on Trend Alignment, Zone Relevance, and Candle Confirmation using weighted algorithms.
- **Zone Clustering:** Uses statistical clustering (via scikit-learn logic) to identify high-probability Supply/Demand zones dynamically.

---

## ⚙️ Tech Stack

### Backend & Infrastructure

| Component | Technology | Role |
|---|---:|---|
| API Framework | FastAPI | Async REST & WebSocket endpoints |
| Task Queue | Celery / message broker | Distributed background processing |
| Message Broker | Redis / PubSub | Pub/Sub, Caching, and Task Brokerage |
| Containerization | Docker / Docker Compose | Service orchestration (core-api vs workers) |
| Reverse Proxy | Nginx / Traefik | Load balancing and routing |

### Data Persistence

| Component | Technology | Role |
|---|---:|---|
| Time Series | InfluxDB / Timescale | Storing OHLCV and high-frequency metric data |
| Relational | Postgres / Supabase | User data, payments, and configuration |

### Data Science & Analytics

| Component | Technology | Role |
|---|---:|---|
| Quant Analysis | Pandas, NumPy, SciPy | Vectorized market data manipulation |
| Indicators | TA-Lib | Technical analysis primitives |
| ML / Clustering | Scikit-Learn | Adaptive thresholding and zone clustering |

---

## 📂 Project Structure

```
Claw/
├── docker/                  # Docker contexts for specific microservices
│   ├── core-api/            # API Gateway container config
│   ├── service-workers/     # Background worker container config
│   ├── influxdb/            # Time-series DB setup
│   └── redis/               # Broker setup
│
├── src/
│   ├── core/
│   │   ├── engines/         # The heavy lifters (PatternEngine, ChartEngine)
│   │   ├── services/        # Business logic (Signals, Notifications)
│   │   └── use_cases/       # Domain logic (Trader-Aware Analysis Pipeline)
│   │       ├── trend_detector.py    # Market structure identification
│   │       ├── scorer.py            # The "ML-ish" weighted scoring logic
│   │       └── pattern_scanner.py   # Distributed scanning logic
│   │
│   ├── presentation/        # API Routes (REST + WebSockets)
│   └── infrastructure/      # DB adapters (Supabase, Influx, Redis)
│
├── telegram/                # Telegram Bot service (standalone capability)
└── requirements.txt         # Production dependencies
```

---

## 🚀 Getting Started

For local iOS development, the app connects to one API and a single launcher
supervises the background processes. Use the existing `.env` and `.venv`.
Open OrbStack / Docker Desktop and start the configured Redis and InfluxDB
services first. On the current Mac these are `claw_redis` and `claw_influxdb`.

```sh
.venv/bin/python scripts/dev_backend.py check
.venv/bin/python scripts/dev_backend.py start --scanner
```

In a second terminal, expose port 8000 using the app's existing tunnel:

```sh
ngrok http --url=stable-wholly-crappie.ngrok-free.app 8000
```

`--scanner` enables the bounded 10-symbol Binance pilot on four intervals.
Startup can require initial candle preparation; catalog definitions alone do
not mean live scans are ready. Alert/push workers are not started. Ctrl-C stops
the launcher's processes. Logs are under `logs/dev-backend/`.

See **[Running the backend for iOS](docs/local-ios-backend.md)** for database
startup, process roles, health checks, iPhone configuration and troubleshooting.
The familiar manual `PYTHONPATH` commands also remain in `src/app.py`.
There is no root Compose file; the older generic `docker-compose up` instruction
did not match this checkout.

---

## 📊 The "Trader-Aware" Engine

This backend features a dedicated system for Trader-Aware Analysis, documented in detail in `TRADER_AWARE_ANALYSIS_README.md`.

**Key Capabilities:**
- **Trend & Swing Analysis:** Auto-detection of HH/HL (Higher Highs/Lows) market structure.
- **Adaptive Scoring:** A setup is only flagged if it meets a dynamic confidence threshold derived from multiple data points.
- **Conflict Resolution:** A hierarchical priority system (Harmonic > Chart > Candlestick) ensures users aren't flooded with conflicting signals.

---

## 🤝 Contributing

1. Fork the repo.
2. Feature Branch: Create a branch for your microservice feature (e.g., `feature/new-worker-logic`).
3. Tests: Ensure `tests/unit` pass.
4. PR: Submit a Pull Request with a description of the architectural changes.

---

## 📜 License

MIT License

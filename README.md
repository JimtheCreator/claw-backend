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

Because this is a distributed system, you are spinning up a mesh of containers rather than a single script.

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local debugging)
- External API Keys (Binance, Telegram, Stripe)

### 2. Environment Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Ensure you populate `REDIS_HOST`, `INFLUXDB_URL`, and your market data provider keys.

### 3. Launching the Cluster

Use Docker Compose to spin up the entire stack (API, Workers, DBs, Cache):

```bash
# Build and start all services in detached mode
docker-compose up --build -d
```

**Services created:**
- `core-api`: Accessible at http://localhost:8000
- `data-workers`: Background logs available via `docker logs -f watchers-data-workers-1`
- `redis`: Port 6379
- `influxdb`: Port 8086

### 4. Running Locally (Development)

If you need to debug the Data Science logic without the full Docker overhead, you can run the analyzer directly:

```bash
# Install specific DS dependencies
pip install pandas numpy scikit-learn ta-lib

# Run the analysis test suite
python tests/integration/test_pattern_detection_workflow.py
```

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

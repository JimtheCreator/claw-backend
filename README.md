# Claw Backend

## Overview
Claw Backend is a Flask-based backend system designed for market trend analysis and trading pattern detection. It integrates Redis, PostgreSQL, Celery, InfluxDB, and Nginx, with Docker handling deployment and service orchestration.

## Technologies Used
- **Flask** - Web framework
- **Redis** - Caching and task queue backend
- **PostgreSQL** - Primary database
- **Celery** - Asynchronous task queue
- **InfluxDB** - Time-series database for storing market data
- **Nginx** - Reverse proxy and load balancing
- **Docker** - Containerization for deployment

## Project Structure
```plaintext
Claw-Backend/
├── .gitignore               # Ignore sensitive & unnecessary files
├── docker-compose.yml       # Orchestrates Postgres, Redis, InfluxDB, etc.
├── Dockerfile               # Flask/Celery app containerization
├── Makefile                 # Shortcut commands for development
├── requirements.txt         # Python dependencies
│
├── src/                     # Main application source code
│   ├── __init__.py
│   ├── app.py               # Flask app factory
│   │
│   ├── core/                # Core business logic
│   │   ├── domain/          # Domain entities
│   │   ├── services/        # Business services
│   │   ├── use_cases/       # Application logic
│   │
│   ├── infrastructure/      # External integrations
│   │   ├── database/        # SQLAlchemy setup & migrations
│   │   ├── data_sources/    # Market data providers
│   │   │   ├── binance.py
│   │   │   ├── yahoofinance.py
│   │   │   ├── tradingview.py
│   │   ├── external_apis/   # Third-party API clients
│   │
│   ├── presentation/        # API & Web Layer
│   │   ├── api/
│   │   │   ├── routes/      # Flask API routes
│   │   │   ├── middleware/   # Authentication, rate limiting
│   │   │   ├── schemas/      # Pydantic models for validation
│   │   ├── web/              # Web-based UI (if applicable)
│   │
│   ├── common/              # Shared utilities
│   │   ├── config.py        # Configuration management
│   │   ├── utils/           # Helper functions
│
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── e2e/                 # End-to-end tests
│   └── conftest.py          # Pytest fixtures
│
├── docker/                  # Docker configurations
│   ├── nginx/
│   │   ├── nginx.conf       # Reverse proxy settings
│   ├── postgres/            # PostgreSQL setup
│   ├── redis/               # Redis setup
│   ├── influxdb/
│   │   ├── init.sh          # DB initialization script
│
├── scripts/                 # Utility scripts
│   ├── db/
│   │   ├── migrations.py    # Database migrations
│   ├── data/
│   │   ├── seed_data.py     # Seed test data
│   │   ├── monitor_tasks.py # Celery task monitoring
│
├── docs/                    # Documentation
│   ├── API.md               # OpenAPI spec
│   ├── patterns.md          # Trading patterns documentation
│
└── .github/                 # CI/CD workflows
    ├── workflows/
    │   ├── main.yml         # GitHub Actions pipeline
```

## Installation & Setup
### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/Claw-Backend.git
cd Claw-Backend
```

### 2. Create a Virtual Environment
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Copy the `.env.example` file and configure necessary settings:
```sh
cp .env.example .env
```

### 5. Run the Application with Docker
```sh
docker-compose up --build
```

## Deployment
To deploy to production, ensure you configure Nginx and database settings properly before running:
```sh
docker-compose -f docker-compose.prod.yml up -d
```

## Contributing
1. Fork the repository
2. Create a new branch (`feature/my-feature`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

**Note:** The `.env` file is ignored via `.gitignore`, so ensure it is manually configured on any deployed instance.


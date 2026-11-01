# 🏎️ F1 AI Telemetry & Race Predictor

[![Production Status](https://img.shields.io/badge/Status-Production%20Live-brightgreen)](https://f1-race-predictor-live-api-data-pipeline.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning telemetry and race outcome prediction pipeline. Powered by **Scikit-Learn Gradient Boosting**, the **Jolpica-F1 Live API**, and 75 years of historical Formula 1 Grand Prix telemetry (1950–2026).

---

## 🌐 Live Production Deployment
- **Live Streamlit Telemetry App:** [https://f1-race-predictor-live-api-data-pipeline.onrender.com/](https://f1-race-predictor-live-api-data-pipeline.onrender.com/)

---

## 🏛️ Production Project Architecture

```text
f1-race-predictor/
├── .github/
│   └── workflows/
│       ├── lint.yml                # CI: Code formatting & flake8 linting
│       └── test.yml                # CI: Automated pytest test suite
├── .streamlit/
│   └── config.toml                 # Cloud networking & Streamlit UI configuration
├── data/                           # Canonical datasets (historical + live snapshots)
│   ├── latest_circuit_stats.csv    # Track characteristics, lap counts, circuit metadata
│   ├── latest_driver_stats.csv     # Driver career metrics, podium % & win rate
│   ├── latest_team_stats.csv       # Constructor baseline performance metrics
│   ├── master_dataset.csv          # Merged multi-decade F1 training corpus
│   └── races.csv                   # Historical Grand Prix schedule & outcomes
├── docs/                           # Technical documentation & architecture guides
│   ├── API_SPECIFICATION.md        # REST API endpoints & request/response schemas
│   └── MODEL_ARCHITECTURE.md       # ML model architectures, validation & metrics
├── models/                         # Serialized ML artifacts & feature pipelines
│   ├── feature_cols.pkl            # Feature column alignment schema
│   ├── podium_model.pkl            # Gradient Boosting Classifier for podium finish
│   ├── position_model.pkl          # Gradient Boosting Regressor for finish position
│   └── winner_model.pkl            # Gradient Boosting Classifier for race win
├── notebooks/                      # Exploratory Data Analysis & Model Training
│   ├── 01_live_api_and_data_pipeline.ipynb
│   └── 02_feature_engineering_and_models.ipynb
├── src/                            # Production Python core library
│   ├── __init__.py
│   ├── api_client.py               # Jolpica-F1 / Ergast Live REST client
│   ├── app_utils.py                # Telemetry UI helpers and caching
│   ├── config.py                   # Centralized configuration & environment loader
│   ├── data_loader.py              # Ingestion, validation, and season filtering
│   ├── evaluator.py                # Validation metrics (AUC, Accuracy, MAE)
│   ├── preprocessor.py             # Feature normalization & engineering pipelines
│   └── features/                   # Modular feature extractors
│       ├── __init__.py
│       ├── circuit_features.py     # Circuit-specific performance scoring
│       ├── driver_features.py      # Rolling form & historical averages
│       ├── reliability_features.py  # DNF and mechanical failure rate modeling
│       └── team_features.py        # Constructor relative pace index
├── static/                         # High-performance custom web interface
│   ├── app.js                      # Interactive telemetry visualizer & live simulator
│   ├── index.html                  # Responsive race dashboard layout
│   └── styles.css                  # Modern glassmorphism & typography
├── tests/                          # Automated unit and integration test suite
│   ├── __init__.py
│   ├── test_api.py                 # Live API client tests
│   ├── test_features.py            # Feature transformation tests
│   └── test_models.py              # Model artifact integrity tests
├── .dockerignore                   # Build context optimization
├── .env.example                    # Environment variable documentation
├── .gitignore                      # Git exclusion rules
├── Makefile                        # Unified developer workflow CLI
├── Dockerfile                      # Production Docker container definition
├── docker-compose.yml              # Local container orchestration
├── pyproject.toml                  # PEP 518/621 package metadata & tooling config
├── render.yaml                     # Render Cloud Infrastructure as Code blueprint
├── requirements.txt                # Pinned production dependencies
├── setup.py                        # Python setuptools packaging
├── app.py                          # Streamlit Telemetry Dashboard Entrypoint
└── main.py                         # FastAPI Production REST Service Entrypoint
```

---

## ⚡ Quick Start

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/preetamvr1595/F1-Race-Predictor-Live-API-Data-Pipeline.git
cd F1-Race-Predictor-Live-API-Data-Pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Applications

**Run the Streamlit Telemetry Dashboard:**
```bash
streamlit run app.py
# Or with Makefile:
make run-app
```

**Run the FastAPI Backend & Web UI:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Or with Makefile:
make run-api
```

---

## 🐳 Docker Deployment

Build and run the container locally:

```bash
docker build -t f1-race-predictor:latest .
docker run -p 8501:8501 -e PORT=8501 f1-race-predictor:latest
```

---

## 🧪 Testing & Code Quality

Run tests:
```bash
pytest tests/ -v
# Or:
make test
```

---

## 📊 Model Performance (2023–2024 Out-of-Time Test Set)

| Model | Target | Evaluation Metric | Score | Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Podium Classifier** | Finish in Top 3 | ROC-AUC / Accuracy | **0.924** / 89.4% | 75.0% |
| **Winner Classifier** | Race 1st Place | ROC-AUC / Accuracy | **0.960** / 95.7% | 90.0% |
| **Position Regressor** | Exact Finish Position | Mean Absolute Error (MAE) | **3.14 positions** | 3.60 naive |

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

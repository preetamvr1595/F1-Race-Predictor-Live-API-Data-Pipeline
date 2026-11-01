.PHONY: help install run-app run-api test lint docker-build docker-run clean

help:
	@echo "F1 AI Race Predictor — Production CLI"
	@echo "--------------------------------------"
	@echo "make install       Install dependencies"
	@echo "make run-app       Start Streamlit Dashboard"
	@echo "make run-api       Start FastAPI Backend & Web UI"
	@echo "make test          Run pytest suite"
	@echo "make lint          Run code linting"
	@echo "make docker-build  Build Docker container image"
	@echo "make docker-run    Run Docker container"
	@echo "make clean         Remove build and cache artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run-app:
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

run-api:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100

docker-build:
	docker build -t f1-race-predictor:latest .

docker-run:
	docker run -p 8501:8501 -e PORT=8501 f1-race-predictor:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache build/ dist/ *.egg-info 2>/dev/null || true

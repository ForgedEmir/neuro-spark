# NeuroSpark — Makefile
# Usage: make help

.PHONY: help up down build download dashboard notebook logs restart status \
        test lint mlflow-ui install-dev clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker ──
up: ## Start the Spark cluster (detached)
	cd "Analyse Cerveau" && docker-compose up -d
	@echo "Services:"
	@echo "  Jupyter:    http://localhost:8889"
	@echo "  Spark UI:   http://localhost:8080"
	@echo "  History:    http://localhost:18080"

down: ## Stop the Spark cluster
	cd "Analyse Cerveau" && docker-compose down

build: ## Rebuild Docker images (cached layers)
	cd "Analyse Cerveau" && docker-compose build

rebuild: ## Full rebuild without cache
	cd "Analyse Cerveau" && docker-compose build --no-cache

logs: ## Show container logs
	cd "Analyse Cerveau" && docker-compose logs -f --tail=50

restart: ## Restart containers
	cd "Analyse Cerveau" && docker-compose restart

status: ## Show container status
	cd "Analyse Cerveau" && docker-compose ps

# ── Data Pipeline ──
download: ## Download EEG dataset (inside container)
	cd "Analyse Cerveau" && python scripts/download_eeg.py

ingest: ## EDF → Parquet (partitionné)
	cd "Analyse Cerveau" && python src/neuro_spark/pipelines/ingestion.py

features: ## FFT band power features
	cd "Analyse Cerveau" && python src/neuro_spark/pipelines/features.py

train: ## CrossValidator + MLflow tracking
	cd "Analyse Cerveau" && python src/neuro_spark/pipelines/training.py

evaluate: ## Metrics
	cd "Analyse Cerveau" && python src/neuro_spark/pipelines/evaluation.py

export: ## Export dashboard data
	cd "Analyse Cerveau" && python src/neuro_spark/pipelines/export_dashboard.py

pipeline: ingest features train evaluate export ## Run full pipeline

# ── Quality ──
test: ## Run test suite
	cd "Analyse Cerveau" && python -m pytest tests/ -v

lint: ## Check code style with ruff
	cd "Analyse Cerveau" && ruff check src/neuro_spark/ src/neuro_spark/pipelines/ dashboard.py scripts/download_eeg.py --ignore E501

lint-fix: ## Auto-fix ruff issues
	cd "Analyse Cerveau" && ruff check src/neuro_spark/ src/neuro_spark/pipelines/ dashboard.py scripts/download_eeg.py --ignore E501 --fix

install-dev: ## Install dev dependencies
	pip install ruff pytest pyyaml

# ── Dashboard ──
dashboard: ## Launch dashboard
	cd "Analyse Cerveau" && python dashboard.py

notebook: ## Open Jupyter
	@echo "Opening http://localhost:8889"
	@xdg-open http://localhost:8889 2>/dev/null || open http://localhost:8889 2>/dev/null || true

mlflow-ui: ## Launch MLflow UI
	@echo "Launching MLflow UI at http://localhost:5000"
	cd "Analyse Cerveau" && mlflow ui --backend-store-uri /opt/spark/mlruns --host 0.0.0.0

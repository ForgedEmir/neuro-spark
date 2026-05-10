# NeuroSpark — Makefile
# Usage: make help

.PHONY: help up down build download dashboard notebook logs restart status \
        run ingest features train evaluate export pipeline \
        test lint lint-fix install-dev clean mlflow-ui

# ── Environnement Spark local ──────────────────────────────────────────────
KEDRO = KEDRO_DISABLE_TELEMETRY=1 \
        SPARK_MASTER=local[2] \
        SPARK_LOCAL_DIR=spark-tmp \
        JAVA_HOME=/usr/lib/jvm/java-17-openjdk \
        PYSPARK_PYTHON=.venv/bin/python \
        PYSPARK_DRIVER_PYTHON=.venv/bin/python \
        MLFLOW_TRACKING_URI=mlruns \
        .venv/bin/python -W ignore -m kedro

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Pipeline Kedro ─────────────────────────────────────────────────────────
run: ## Lancer le pipeline complet (Kedro)
	$(KEDRO) run

ingest: ## EDF → Parquet
	$(KEDRO) run --pipeline ingestion

features: ## FFT band power features
	$(KEDRO) run --pipeline features

train: ## CrossValidator + MLflow tracking
	$(KEDRO) run --pipeline training

evaluate: ## Métriques sur le test set
	$(KEDRO) run --pipeline evaluation

export: ## Export données dashboard
	$(KEDRO) run --pipeline export

pipeline: ingest features train evaluate export ## Pipeline complet étape par étape

# ── Données ────────────────────────────────────────────────────────────────
download: ## Télécharger le dataset EEG
	.venv/bin/python scripts/download_eeg.py

# ── Dashboard & UI ─────────────────────────────────────────────────────────
dashboard: ## Lancer le dashboard Dash
	DASHBOARD_DIR=data/dashboard METRICS_DIR=data/metrics PREDICTIONS_DIR=data/predictions \
	.venv/bin/python dashboard.py

mlflow-ui: ## Ouvrir MLflow UI (http://localhost:5000)
	@echo "MLflow UI → http://localhost:5000"
	MLFLOW_TRACKING_URI=mlruns .venv/bin/mlflow ui

kedro-viz: ## Visualiser le pipeline Kedro (http://localhost:4141)
	@echo "Kedro Viz → http://localhost:4141"
	KEDRO_DISABLE_TELEMETRY=1 .venv/bin/python -W ignore -m kedro viz run --port 4141

notebook: ## Ouvrir Jupyter
	@xdg-open http://localhost:8889 2>/dev/null || true

# ── Docker ─────────────────────────────────────────────────────────────────
up: ## Démarrer le cluster Spark Docker
	docker-compose up -d
	@echo "Spark UI  → http://localhost:8080"
	@echo "Jupyter   → http://localhost:8889"
	@echo "History   → http://localhost:18080"

down: ## Arrêter le cluster
	docker-compose down

build: ## Build l'image Docker
	docker-compose build

logs: ## Logs des conteneurs
	docker-compose logs -f --tail=50

# ── Qualité ────────────────────────────────────────────────────────────────
test: ## Tests unitaires
	.venv/bin/python -m pytest tests/ -v

lint: ## Vérifier le style (ruff)
	.venv/bin/ruff check src/ dashboard.py scripts/ --ignore E501

lint-fix: ## Corriger le style automatiquement
	.venv/bin/ruff check src/ dashboard.py scripts/ --ignore E501 --fix

install-dev: ## Installer les dépendances de dev
	.venv/bin/pip install ruff pytest

clean: ## Supprimer les fichiers temporaires Spark
	rm -rf spark-tmp/

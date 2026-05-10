# NeuroSpark — Makefile

.PHONY: help \
        run ingest features train evaluate export pipeline \
        download dashboard kedro-viz mlflow-ui mlflow-report \
        dvc-repro dvc-status dvc-dag \
        test lint lint-fix install-dev clean

# Environnement Spark local (master, JVM, Python du venv, MLflow SQLite)
KEDRO = KEDRO_DISABLE_TELEMETRY=1 \
        SPARK_MASTER=local[2] \
        SPARK_LOCAL_DIR=spark-tmp \
        JAVA_HOME=/usr/lib/jvm/java-17-openjdk \
        PYSPARK_PYTHON=.venv/bin/python \
        PYSPARK_DRIVER_PYTHON=.venv/bin/python \
        MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db \
        .venv/bin/python -W ignore -m kedro

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Pipeline Kedro
run: ## Lancer la pipeline complete
	$(KEDRO) run

ingest: ## Convertir EDF en Parquet
	$(KEDRO) run --pipeline ingestion

features: ## Extraire les features FFT
	$(KEDRO) run --pipeline features

train: ## Entrainer le modele avec CrossValidator
	$(KEDRO) run --pipeline training

evaluate: ## Evaluer le modele sur le test set
	$(KEDRO) run --pipeline evaluation

export: ## Exporter les donnees pour le dashboard
	$(KEDRO) run --pipeline export

pipeline: ingest features train evaluate export ## Pipeline etape par etape

# Donnees
download: ## Telecharger le dataset EEG depuis PhysioNet
	.venv/bin/python scripts/download_eeg.py

# Dashboard et UIs
dashboard: ## Lancer le dashboard Dash
	DASHBOARD_DIR=data/dashboard METRICS_DIR=data/metrics PREDICTIONS_DIR=data/predictions \
	.venv/bin/python dashboard.py

mlflow-ui: ## Ouvrir MLflow UI sur http://localhost:5000
	@echo "MLflow UI -> http://localhost:5000"
	.venv/bin/mlflow ui \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root mlruns \
		--host 127.0.0.1 --port 5000

mlflow-report: ## Generer un rapport graphique comparant les runs MLflow
	MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db .venv/bin/python scripts/mlflow_report.py

kedro-viz: ## Visualiser le DAG Kedro sur http://localhost:4141
	@echo "Kedro Viz -> http://localhost:4141"
	KEDRO_DISABLE_TELEMETRY=1 .venv/bin/python -W ignore -m kedro viz run --port 4141

# DVC (versioning des donnees + reexecution selective)
dvc-repro: ## Rejouer la pipeline DVC, en sautant les etapes deja calculees
	.venv/bin/dvc repro

dvc-status: ## Voir l etat des stages DVC
	.venv/bin/dvc status

dvc-dag: ## Afficher le graphe des stages DVC
	.venv/bin/dvc dag

# Qualite
test: ## Lancer les tests unitaires
	.venv/bin/python -m pytest tests/ -v

lint: ## Verifier le style avec ruff
	.venv/bin/ruff check src/ dashboard.py scripts/ --ignore E501

lint-fix: ## Corriger le style automatiquement
	.venv/bin/ruff check src/ dashboard.py scripts/ --ignore E501 --fix

install-dev: ## Installer les outils de dev
	.venv/bin/pip install ruff pytest

clean: ## Supprimer les fichiers temporaires Spark
	rm -rf spark-tmp/

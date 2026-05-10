# NeuroSpark — Contributing Guide

## Setup

```bash
git clone https://github.com/ForgedEmir/neuro-spark.git
cd neuro-spark/Analyse\ Cerveau

# Install dev dependencies
pip install ruff pytest pyyaml
```

## Code Style

- **Ruff** pour le linting : `ruff check . --ignore E501`
- **Type hints** sur toutes les fonctions publiques
- **Docstrings** en français pour les fonctions métier

## Tests

```bash
cd Analyse\ Cerveau
python -m pytest tests/ -v
```

## Pipeline

```bash
cd Analyse\ Cerveau
docker-compose up -d         # Start Spark cluster
python src/neuro_spark/pipelines/ingestion.py    # EDF → Parquet
python src/neuro_spark/pipelines/features.py     # FFT features
python src/neuro_spark/pipelines/training.py     # CrossValidator
python src/neuro_spark/pipelines/evaluation.py   # Metrics
python src/neuro_spark/pipelines/export_dashboard.py  # Dashboard data
python dashboard.py          # Launch UI → :8050
```

## PR Checklist

- [ ] Ruff check passe
- [ ] Tests passent
- [ ] Type hints présents
- [ ] Docstrings à jour

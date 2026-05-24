# Contributing to NeuroSpark

Thanks for your interest.

## Quick Start

```bash
git clone https://github.com/ForgedEmir/neuro-spark.git
cd neuro-spark
# Docker recommended for Spark environment
docker compose up --build
```

## What's Helpful

- **Streaming improvements** — the Spark Structured Streaming pipeline for EEG is a work in progress.
- **New features** — additional frequency bands, model architectures, or visualization modes.
- **Tests** — more edge cases for the Kedro pipelines.
- **Docs** — architecture diagrams, troubleshooting guides.

## PR Guidelines

1. Branch from `develop`. Name: `feat/description` or `fix/description`.
2. One change per PR.
3. Run `make lint` before pushing.
4. Update the DVC pipeline if data dependencies change.

## Code Style

- Python: ruff, line length 120 (E501 ignored).
- Docstrings in French for business logic, English for technical code.
- No hardcoded paths — use `conf/base/catalog.yml` and environment variables.

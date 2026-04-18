# NeuroSpark — Makefile
# Usage: make help

.PHONY: help up down build download dashboard notebook logs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start the Spark cluster
	cd "Analyse Cerveau" && docker-compose up -d
	@echo "\nServices:"
	@echo "  Jupyter:    http://localhost:8889"
	@echo "  Spark UI:   http://localhost:8080"
	@echo "  History:    http://localhost:18080"

down: ## Stop the Spark cluster
	cd "Analyse Cerveau" && docker-compose down

build: ## Rebuild Docker images (no cache)
	cd "Analyse Cerveau" && docker-compose build --no-cache

download: ## Download EEG dataset (run inside container)
	python "Analyse Cerveau/scripts/download_eeg.py"

dashboard: ## Launch the dashboard
	python "Analyse Cerveau/dashboard.py"

notebook: ## Open Jupyter in browser
	@echo "Opening http://localhost:8889"
	@xdg-open http://localhost:8889 2>/dev/null || open http://localhost:8889 2>/dev/null || true

logs: ## Show container logs
	cd "Analyse Cerveau" && docker-compose logs -f --tail=50

restart: ## Restart all containers
	cd "Analyse Cerveau" && docker-compose restart

status: ## Show container status
	cd "Analyse Cerveau" && docker-compose ps

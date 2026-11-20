# make help lists the targets.
.PHONY: help install test up train stream replay drift api advisor dashboard down

help: ## Show this list
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | sed -E "s/:.*## /\t/" | expand -t 12

install: ## Install everything for local development (Python 3.11 and Java 17)
	pip install -r requirements/dev.txt

test: ## Unit tests (no services needed)
	python -m pytest -q

up: ## Start Redpanda, Postgres, MLflow, then train a champion, then the rest
	docker compose up -d --wait redpanda postgres mlflow
	docker compose run --rm trainer
	docker compose up -d api stream monitor prometheus grafana

train: ## Train, validate and register a new model version (promotes it if it passes)
	docker compose run --rm trainer python -m abslive.train --train-end 2026-09-30

replay: ## Stream the second half of the season into Redpanda at 300 pitches a second
	docker compose run --rm producer

drift: ## Replay one day with a synthetic 3-inch shift, to see the drift alert fire
	docker compose run --rm producer python -m stream.producer --start 2026-09-01 --days 1 --rate 0 \
		--drift-from 2026-09-01 --drift-shift 3

api: ## Run the API locally from model/, without the stack
	uvicorn service.app:app --reload --port 8000

advisor: ## Rebuild web/advisor_data.js from the current champion
	docker compose run --rm -v "$(CURDIR)/web:/out" producer python -m abslive.export_advisor --out /out/advisor_data.js

dashboard: ## Regenerate the Grafana dashboard JSON from infra/grafana/build_dashboard.py
	python infra/grafana/build_dashboard.py

down: ## Stop everything and delete its data
	docker compose --profile jobs down -v

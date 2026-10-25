.PHONY: help install dq pipeline train test run assistant docker-build docker-up docker-down

help:
	@echo "Supply Chain Intelligence Mini-Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       Install python dependencies"
	@echo "  make dq            Run standalone data quality checks against raw CSVs"
	@echo "  make pipeline      Run layered idempotent ETL pipeline (Raw -> Curated -> Serving)"
	@echo "  make train         Train ML delay risk prediction model"
	@echo "  make test          Run full unit and integration test suite (pytest)"
	@echo "  make run           Start local FastAPI development server"
	@echo "  make assistant     Run the GenAI Assistant CLI demo walkthrough"
	@echo "  make docker-build  Build docker container image"
	@echo "  make docker-up     Start full stack with docker compose"
	@echo "  make docker-down   Stop docker compose stack"

install:
	pip install -r requirements.txt

dq:
	python dq_check.py --input ./data/raw --output ./reports/dq_report.json

pipeline:
	python -m src.data.pipeline --raw-dir ./data/raw --db-path ./data/curated/supply_chain.duckdb

train:
	python -m src.ml.train --db-path ./data/curated/supply_chain.duckdb --output-dir ./models

test:
	pytest tests/ -v

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

assistant:
	python run_assistant.py --demo

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

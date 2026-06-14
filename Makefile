.PHONY: help install test lint format docs clean dashboard api docker-build docker-run

help:
	@echo "AREval - Agent Regression Evaluation Harness"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install all dependencies"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run linters (ruff, mypy)"
	@echo "  make format       Format code (black)"
	@echo "  make dashboard    Start development dashboard"
	@echo "  make api          Start API server"
	@echo "  make docker-build Build Docker images"
	@echo "  make docker-run   Run with Docker Compose"
	@echo "  make clean        Clean build artifacts"

install:
	pip install -e ".[all,dev]"
	cd areval-dashboard && npm install

test:
	pytest tests/ -v --cov=areval --cov-report=term-missing

lint:
	ruff check areval-engine/areval areval-sdk/areval_sdk areval-cli/areval_cli
	mypy areval-engine/areval

format:
	black areval-engine/ areval-sdk/ areval-cli/ tests/

dashboard:
	cd areval-dashboard && npm run dev

api:
	cd areval-api && uvicorn areval_api.main:app --reload --port 8000

docker-build:
	docker-compose build

docker-run:
	docker-compose up

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf areval-dashboard/.next areval-dashboard/dist areval-dashboard/node_modules

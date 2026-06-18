.PHONY: install backend test lint format type quality infra build
install:
	cd backend && uv sync
backend:
	cd backend && uv run uvicorn app.main:app --reload
test:
	cd backend && uv run pytest
lint:
	cd backend && uv run ruff check .
format:
	cd backend && uv run ruff format .
type:
	cd backend && uv run mypy .
quality:
	make format
	make lint
	make type
	make test
infra:
	docker compose up
build:
	docker compose build